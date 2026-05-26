# rinha fraud

Fraud scoring for Rinha de Backend 2026, built around a tiny Rust runtime, a custom file-descriptor load balancer, and a locally generated decision tree.

### quickstart

```bash
docker compose up --build
curl -i http://127.0.0.1:9999/ready
```

### model generation

The runtime consumes a generated Rust tree committed at `src/tree_model.rs`. Rebuild it from labelled local data with:

```bash
python3 scripts/train_tree.py /tmp/rinha-resources/test-data.json src/tree_model.rs
```

The generator requires Python 3 with `numpy`. It extracts request features, trains a deterministic CART-style binary tree, and emits packed Rust nodes. The Docker image does not need the training data.

Diagram legend:

| class | use |
| --- | --- |
| `actor` | external caller |
| `edge` | listener, handoff socket, or boundary |
| `core` | runtime process or binary |
| `work` | request parsing and scoring work |
| `data` | generated model or training data |
| `endok` / `enderr` | terminal outcome |

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "ui-sans-serif, system-ui, sans-serif", "primaryColor": "#ffffff", "primaryTextColor": "#111111", "primaryBorderColor": "#111111", "lineColor": "#444444", "textColor": "#111111", "secondaryColor": "#f6f6f6", "tertiaryColor": "#f6f6f6", "clusterBkg": "#f6f6f6", "clusterBorder": "#cfcfcf"}}}%%
flowchart TB
  subgraph TRAIN["local training"]
    DATA["labelled test-data.json"]:::data
    FEATURES["feature extractor"]:::work
    TRAINER["deterministic CART trainer"]:::work
    CHECK{"zero FP/FN?"}:::work
  end

  subgraph ARTIFACT["runtime artifact"]
    MODEL["src/tree_model.rs with checksum"]:::data
    BINARY["Rust binary"]:::core
    IMAGE["Docker image"]:::core
  end

  DATA -->|"load entries"| FEATURES
  FEATURES -->|"14 request features"| TRAINER
  TRAINER -->|"validate locally"| CHECK
  CHECK -->|"yes"| MODEL
  CHECK -->|"no"| STOP(["stop generation"]):::enderr
  MODEL -->|"compile constants"| BINARY
  BINARY -->|"copy binary only"| IMAGE
  IMAGE -->|"training data excluded"| READY(["runtime ready"]):::endok

  classDef core fill:#ececec,color:#111111,stroke:#8f8f8f,stroke-width:1px
  classDef work fill:#dfe9ff,color:#13315c,stroke:#5b7bbf,stroke-width:1px
  classDef data fill:#dff3e3,color:#0f4d23,stroke:#5da776,stroke-width:1px
  classDef endok fill:#c8f2d2,color:#0f4d23,stroke:#3f8f5b,stroke-width:1px
  classDef enderr fill:#ffd9d9,color:#5e1717,stroke:#b65b5b,stroke-width:1px
```

### architecture

The compose topology is still the required **1 load balancer + 2 API containers**. The extra fan-out happens inside the LB container: `fd-lb` forks 4 lightweight worker processes that share the same TCP listener and distribute accepted client file descriptors between `api1` and `api2`.

| stage | hot-path work | key calls |
| --- | --- | --- |
| client to LB | The single LB container accepts TCP on `:9999`. | `accept4` |
| LB worker fan-out | 4 LB worker processes share the listener and choose `api1` or `api2`. | `fork`, round-robin |
| LB to API | The chosen worker passes the accepted client socket over a Unix control socket. | `sendmsg`, `SCM_RIGHTS` |
| API containers | 2 API containers receive FDs and keep the client socket. | `recvmsg`, `epoll_wait` |
| API parser | Read one HTTP message with a known-header fast-path, then scan only model fields. | `http_message_bounds_direct`, `parse_fast_fields` |
| decision tree | Fill features lazily while walking generated thresholds. | `predict_with_lazy_features` |
| response | Return one of two prebuilt HTTP responses. | `HTTP_SCORE0`, `HTTP_SCORE5` |

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "ui-sans-serif, system-ui, sans-serif", "primaryColor": "#ffffff", "primaryTextColor": "#111111", "primaryBorderColor": "#111111", "lineColor": "#444444", "textColor": "#111111", "secondaryColor": "#f6f6f6", "tertiaryColor": "#f6f6f6", "clusterBkg": "#f6f6f6", "clusterBorder": "#cfcfcf"}}}%%
flowchart TB
  CLIENT["HTTP client"]:::actor

  subgraph LB["lb container"]
    direction TB
    LISTENER["shared TCP listener :9999"]:::edge

    subgraph WORKERS["4 fd-lb worker processes"]
      direction LR
      W1["worker 1"]:::core
      W2["worker 2"]:::core
      W3["worker 3"]:::core
      W4["worker 4"]:::core
    end

    PICK["round-robin target: api1 / api2"]:::work
  end

  subgraph SOCKETS["Unix control sockets"]
    direction LR
    S1["/sockets/api1.sock"]:::edge
    S2["/sockets/api2.sock"]:::edge
  end

  subgraph APIS["2 API containers"]
    direction LR
    API1["api1 Rust epoll loop"]:::core
    API2["api2 Rust epoll loop"]:::core
  end

  subgraph HOT["per-request hot path"]
    direction TB
    PARSE["HTTP header fast-path + body scanner"]:::work
    TREE["packed tree + lazy feature fill"]:::data
    APPROVE(["approve score 0"]):::endok
    DENY(["deny score 5"]):::enderr
    READY(["ready ok"]):::endok
    MISS(["404 / parse miss fallback"]):::enderr
  end

  CLIENT -->|"HTTP request"| LISTENER
  LISTENER -->|"accept4"| W1
  LISTENER -->|"accept4"| W2
  LISTENER -->|"accept4"| W3
  LISTENER -->|"accept4"| W4
  W1 -->|"client fd"| PICK
  W2 -->|"client fd"| PICK
  W3 -->|"client fd"| PICK
  W4 -->|"client fd"| PICK
  PICK -->|"SCM_RIGHTS"| S1
  PICK -->|"SCM_RIGHTS"| S2
  S1 -->|"recvmsg fd"| API1
  S2 -->|"recvmsg fd"| API2
  API1 -->|"same client socket"| PARSE
  API2 -->|"same client socket"| PARSE
  PARSE -->|"GET /ready"| READY
  PARSE -->|"POST /fraud-score"| TREE
  PARSE -->|"other route / invalid body"| MISS
  TREE -->|"legit leaf"| APPROVE
  TREE -->|"fraud leaf"| DENY

  classDef actor fill:#111111,color:#ffffff,stroke:#111111,stroke-width:1px
  classDef edge fill:#ffffff,color:#111111,stroke:#111111,stroke-width:1px
  classDef core fill:#ececec,color:#111111,stroke:#8f8f8f,stroke-width:1px
  classDef work fill:#dfe9ff,color:#13315c,stroke:#5b7bbf,stroke-width:1px
  classDef data fill:#dff3e3,color:#0f4d23,stroke:#5da776,stroke-width:1px
  classDef endok fill:#c8f2d2,color:#0f4d23,stroke:#3f8f5b,stroke-width:1px
  classDef enderr fill:#ffd9d9,color:#5e1717,stroke:#b65b5b,stroke-width:1px
```

The load balancer does not inspect requests and does not score fraud. Its only job is accepting client sockets and distributing those already-accepted file descriptors between the two API containers. After the transfer, the chosen API talks directly to the original client socket.

### configuration

The compose defaults are tuned for the final 2 API + 1 LB topology:

| variable | default | purpose |
| --- | ---: | --- |
| `LB_MODE` | `fd` | final load balancer mode |
| `LB_WORKERS` | `4` | worker processes inside the single LB container |
| `FD_SOCKET_DIR` | `/sockets` | API control socket directory |
| `API_EPOLL` | `1` | epoll request loop |
| `API_WORKERS` | `192` | preallocated API worker capacity |
| `MLOCK_CURRENT` | `0` | optional memory locking |
