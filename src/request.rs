use serde::Deserialize;
use smallvec::SmallVec;

#[derive(Deserialize)]
pub struct FraudRequest<'a> {
    #[serde(borrow)]
    pub transaction: Transaction<'a>,
    #[serde(borrow)]
    pub customer: Customer<'a>,
    #[serde(borrow)]
    pub merchant: Merchant<'a>,
    pub terminal: Terminal,
    #[serde(borrow)]
    pub last_transaction: Option<LastTransaction<'a>>,
}

#[derive(Deserialize)]
pub struct Transaction<'a> {
    pub amount: f64,
    pub installments: u16,
    pub requested_at: &'a str,
}

#[derive(Deserialize)]
pub struct Customer<'a> {
    pub avg_amount: f64,
    pub tx_count_24h: u16,
    #[serde(borrow)]
    pub known_merchants: SmallVec<[&'a str; 4]>,
}

#[derive(Deserialize)]
pub struct Merchant<'a> {
    pub id: &'a str,
    pub mcc: &'a str,
    pub avg_amount: f64,
}

#[derive(Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Deserialize)]
pub struct LastTransaction<'a> {
    pub timestamp: &'a str,
    pub km_from_current: f64,
}
