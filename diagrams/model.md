# Model danych GenericBank (landing / bronze)

```mermaid
erDiagram
    CUSTOMERS ||--o{ ACCOUNTS : "has"
    ACCOUNTS  ||--o{ TRANSACTIONS : "has"

    CUSTOMERS {
        string customer_id PK
        string segment "RETAIL / PREMIER"
        string birth_year
        string region "voivodeship"
        string relationship_start
    }
    ACCOUNTS {
        string account_id PK
        string customer_id FK
        string account_type "CURRENT / SAVINGS / CARD"
        string currency "PLN"
        string open_date
    }
    TRANSACTIONS {
        string transaction_id PK
        string account_id FK
        string txn_ts
        string amount
        string direction "CREDIT / DEBIT"
        string txn_type "SALARY, CARD_PAYMENT, TRANSFER, CASH_DEPOSIT, CASH_WITHDRAWAL"
        string channel "MOBILE, WEB, BRANCH, ATM, POS"
        string merchant_category "only for CARD_PAYMENT"
        string counterparty_country
    }
```

Poza modelem: `ground_truth.csv` (customer_id, pattern, pattern_start): „prawda” o wstrzykniętych wzorcach, tylko do walidacji.
W bronze każda tabela ma dodatkowo `_source_file` i `_ingested_at`.
