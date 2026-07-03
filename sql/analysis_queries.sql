-- Flow matrix straight from SQL: share of each state's items per hub
SELECT customer_state,
       hub,
       ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER (PARTITION BY customer_state), 3)
           AS share
FROM order_items
GROUP BY customer_state, hub
ORDER BY customer_state, share DESC;

-- Weekly demand with a 4-week moving average per state x product group
SELECT week, customer_state, product_group, demand,
       ROUND(AVG(demand) OVER (
           PARTITION BY customer_state, product_group
           ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW), 1)
           AS demand_ma4
FROM weekly_demand
ORDER BY customer_state, product_group, week;

-- Freight paid vs items shipped, by hub (cost driver check)
SELECT hub,
       COUNT(*)                       AS items,
       ROUND(AVG(freight_value), 2)   AS avg_freight_brl,
       ROUND(SUM(freight_value), 0)   AS total_freight_brl
FROM order_items
GROUP BY hub
ORDER BY items DESC;
