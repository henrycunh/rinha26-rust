# rinha fraud

A compact Rust fraud scoring service built around a custom file descriptor load balancer, a hand-written request scanner, deterministic fast-path rules, and a generated residual decision tree.

## quickstart

```bash
docker compose up --build
```

## approach

The hot path is intentionally split into a cheap deterministic layer and a compact learned layer.

The deterministic layer scans only the fields needed for scoring and immediately handles obvious safe or risky requests. These checks are simple business-shaped predicates: familiar merchants, distance, purchase size, recent activity, and merchant category risk.

Requests that are not decided by those rules are converted into a fixed feature vector and passed to a residual decision tree. The tree is trained only on the requests left behind by the deterministic layer, so it focuses on the ambiguous cases instead of relearning the easy ones.

The tree stays as packed data plus a small indexed loop. A fully expanded branch tree was tested and rejected because the larger instruction footprint hurt tail latency.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "ui-sans-serif, system-ui, sans-serif"}}}%%
flowchart TB
  REQUEST["http body"]:::actor

  subgraph parser["runtime parser"]
    SCAN["field scanner"]:::work
    FIELDS["minimal scoring fields"]:::work
  end

  subgraph rules["deterministic fast-path"]
    SAFE{"obviously safe?"}:::work
    RISKY{"obviously risky?"}:::work
  end

  subgraph residual["residual tree"]
    FEATURES["fixed feature vector"]:::data
    MODEL["packed tree data"]:::data
    WALK["small indexed predictor"]:::work
  end

  APPROVE(["approve"]):::endok
  DENY(["deny"]):::enderr

  REQUEST --> SCAN
  SCAN --> FIELDS
  FIELDS --> SAFE
  SAFE -->|"yes"| APPROVE
  SAFE -->|"no"| RISKY
  RISKY -->|"yes"| DENY
  RISKY -->|"no"| FEATURES
  FEATURES --> MODEL
  MODEL --> WALK
  WALK -->|"legit leaf"| APPROVE
  WALK -->|"fraud leaf"| DENY

  classDef actor fill:black,color:white,stroke:black
  classDef work fill:lightsteelblue,color:midnightblue,stroke:steelblue
  classDef data fill:honeydew,color:darkgreen,stroke:seagreen
  classDef endok fill:palegreen,color:darkgreen,stroke:seagreen
  classDef enderr fill:mistyrose,color:maroon,stroke:indianred
```

## model generation

The model is generated offline and committed as Rust source. Generation applies the same deterministic fast-path used at runtime, trains only on the remaining requests, validates exact classification on the training payload, and emits packed nodes for the runtime predictor.

The container does not need training data. It only ships the compiled scorer.

## topology

The load balancer accepts client connections and passes accepted sockets to API peers through Unix control sockets. After the handoff, the selected API process talks directly to the original client socket.

The load balancer does not inspect requests and does not score fraud. Its job is only accepting sockets and distributing them cheaply.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "ui-sans-serif, system-ui, sans-serif"}}}%%
flowchart TB
  CLIENT["http client"]:::actor

  subgraph lb["load balancer container"]
    LISTENER["tcp listener"]:::edge
    WORKER["fd handoff worker"]:::core
    PICK["round-robin socket picker"]:::work
  end

  subgraph controls["unix control sockets"]
    direction LR
    PRIMARY_SOCKET["primary api socket"]:::edge
    SECONDARY_SOCKET["secondary api socket"]:::edge
  end

  subgraph apis["api containers"]
    direction LR
    PRIMARY_API["primary epoll loop"]:::core
    SECONDARY_API["secondary epoll loop"]:::core
  end

  subgraph hotpath["request hot path"]
    PARSE["header fast-path and body scanner"]:::work
    FAST["safe or risky fast-path"]:::work
    TREE["residual tree scorer"]:::data
    RESPONSE(["prebuilt response"]):::endok
    FALLBACK(["fallback response"]):::enderr
  end

  CLIENT -->|"request"| LISTENER
  LISTENER -->|"accept"| WORKER
  WORKER --> PICK
  PICK -->|"fd handoff"| PRIMARY_SOCKET
  PICK -->|"fd handoff"| SECONDARY_SOCKET
  PRIMARY_SOCKET -->|"receive fd"| PRIMARY_API
  SECONDARY_SOCKET -->|"receive fd"| SECONDARY_API
  PRIMARY_API -->|"original socket"| PARSE
  SECONDARY_API -->|"original socket"| PARSE
  PARSE -->|"health check"| RESPONSE
  PARSE -->|"score request"| FAST
  PARSE -->|"invalid request"| FALLBACK
  FAST -->|"resolved"| RESPONSE
  FAST -->|"ambiguous"| TREE
  TREE --> RESPONSE

  classDef actor fill:black,color:white,stroke:black
  classDef edge fill:white,color:black,stroke:black
  classDef core fill:gainsboro,color:black,stroke:gray
  classDef work fill:lightsteelblue,color:midnightblue,stroke:steelblue
  classDef data fill:honeydew,color:darkgreen,stroke:seagreen
  classDef endok fill:palegreen,color:darkgreen,stroke:seagreen
  classDef enderr fill:mistyrose,color:maroon,stroke:indianred
```

## runtime tuning

The compose defaults favor a little more budget for the socket handoff path and keep the load balancer simple. Extra load balancer workers, datagram handoff, and memory locking were tried locally and did not stay in the default configuration.

The remaining optimization target is mostly transport tail behavior rather than scoring logic.
