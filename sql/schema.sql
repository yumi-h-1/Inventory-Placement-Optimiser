-- Optional: load the processed planning tables into PostgreSQL
CREATE TABLE IF NOT EXISTS order_items (
    week            DATE,
    customer_state  VARCHAR(2),
    hub             VARCHAR(20),
    product_group   VARCHAR(20),
    price           NUMERIC,
    freight_value   NUMERIC
);

CREATE TABLE IF NOT EXISTS weekly_demand (
    week            DATE,
    customer_state  VARCHAR(2),
    product_group   VARCHAR(20),
    demand          INTEGER
);
