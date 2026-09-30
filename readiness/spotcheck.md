# Task spot-check (H5)

Mark each task OK / ISSUE. An ISSUE blocks the pilot until the generator is fixed.

## F7-100 (F7-100-rel-dev-s1000)

### F7-100-rel-dev-s1000-t054  [ ] OK  [ ] ISSUE

**Prompt:** Case C-517082: customer CU-55827 has submitted a fraud-review request for $483. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "deny", "approver": "none", "deadline_days": 1, "document": "delivery_photo"}`

**Setup:** `{"customers": {"CU-55827": {"region": "UK", "tier": "platinum"}}}`

**Gold facts:**
- `f-P-9446`: Policy P-9446 (Fraud Reviews, UK region). For a fraud-review request from a customer in the UK region with an amount of at least $250 and under $500, the agent must escalate the request to the Tier-2 supervisor, respond within 4 business days, and the customer must provide a government photo ID.
- `f-X-424`: Exception X-424 (amends Policy P-9446). When the customer holds platinum status, Policy P-9446 does not apply as written: the agent must instead deny the request, respond within 1 business day, and the customer must provide a photo of the delivered package.

### F7-100-rel-dev-s1000-t024  [ ] OK  [ ] ISSUE

**Prompt:** Case C-983262: customer CU-31394 has submitted a fraud-review request for $460. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "escalate", "approver": "tier2_supervisor", "deadline_days": 4, "document": "photo_id"}`

**Setup:** `{"customers": {"CU-31394": {"region": "UK", "tier": "silver"}}}`

**Gold facts:**
- `f-P-9446`: Policy P-9446 (Fraud Reviews, UK region). For a fraud-review request from a customer in the UK region with an amount of at least $250 and under $500, the agent must escalate the request to the Tier-2 supervisor, respond within 4 business days, and the customer must provide a government photo ID.
- `f-X-424`: Exception X-424 (amends Policy P-9446). When the customer holds platinum status, Policy P-9446 does not apply as written: the agent must instead deny the request, respond within 1 business day, and the customer must provide a photo of the delivered package.

### F7-100-rel-dev-s1000-t048  [ ] OK  [ ] ISSUE

**Prompt:** Case C-995648: customer CU-68848 has submitted a fraud-review request for $373. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "deny", "approver": "none", "deadline_days": 1, "document": "delivery_photo"}`

**Setup:** `{"customers": {"CU-68848": {"region": "UK", "tier": "platinum"}}}`

**Gold facts:**
- `f-P-9446`: Policy P-9446 (Fraud Reviews, UK region). For a fraud-review request from a customer in the UK region with an amount of at least $250 and under $500, the agent must escalate the request to the Tier-2 supervisor, respond within 4 business days, and the customer must provide a government photo ID.
- `f-X-424`: Exception X-424 (amends Policy P-9446). When the customer holds platinum status, Policy P-9446 does not apply as written: the agent must instead deny the request, respond within 1 business day, and the customer must provide a photo of the delivered package.

### F7-100-rel-dev-s1000-t056  [ ] OK  [ ] ISSUE

**Prompt:** Case C-847795: customer CU-34401 has submitted a fraud-review request for $93. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "escalate", "approver": "regional_manager", "deadline_days": 2, "document": "delivery_photo"}`

**Setup:** `{"customers": {"CU-34401": {"region": "LATAM", "tier": "silver"}}}`

**Gold facts:**
- `f-P-2659`: Policy P-2659 (Fraud Reviews, LATAM region). For a fraud-review request from a customer in the LATAM region with an amount of at least $50 and under $100, the agent must escalate the request to the regional manager, respond within 2 business days, and the customer must provide a photo of the delivered package.

### F7-100-rel-dev-s1000-t026  [ ] OK  [ ] ISSUE

**Prompt:** Case C-153138: customer CU-31113 has submitted a damaged-goods request for $18,594. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "approve", "approver": "none", "deadline_days": 6, "document": "none"}`

**Setup:** `{"customers": {"CU-31113": {"region": "LATAM", "tier": "platinum"}}}`

**Gold facts:**
- `f-P-7812`: Policy P-7812 (Damaged Goods, LATAM region). For a damaged-goods request from a customer in the LATAM region with an amount of at least $10,000 and under $25,000, the agent must deny the request, respond within 4 business days, and the customer must provide a government photo ID.
- `f-X-761`: Exception X-761 (amends Policy P-7812). When the customer holds platinum status, Policy P-7812 does not apply as written: the agent must instead approve the request, respond within 6 business days, and no supporting document is required.

### F7-100-rel-dev-s1000-t002  [ ] OK  [ ] ISSUE

**Prompt:** Case C-318326: customer CU-77616 has submitted a damaged-goods request for $13,321. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "approve", "approver": "none", "deadline_days": 6, "document": "none"}`

**Setup:** `{"customers": {"CU-77616": {"region": "LATAM", "tier": "platinum"}}}`

**Gold facts:**
- `f-P-7812`: Policy P-7812 (Damaged Goods, LATAM region). For a damaged-goods request from a customer in the LATAM region with an amount of at least $10,000 and under $25,000, the agent must deny the request, respond within 4 business days, and the customer must provide a government photo ID.
- `f-X-761`: Exception X-761 (amends Policy P-7812). When the customer holds platinum status, Policy P-7812 does not apply as written: the agent must instead approve the request, respond within 6 business days, and no supporting document is required.

### F7-100-rel-dev-s1000-t016  [ ] OK  [ ] ISSUE

**Prompt:** Case C-908568: customer CU-54020 has submitted a shipping-delay compensation request for $7,834. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "deny", "approver": "none", "deadline_days": 3, "document": "photo_id"}`

**Setup:** `{"customers": {"CU-54020": {"region": "UK", "tier": "gold"}}}`

**Gold facts:**
- `f-P-8344`: Policy P-8344 (Shipping Delays, UK region). For a shipping-delay compensation request from a customer in the UK region with an amount of at least $5,000 and under $10,000, the agent must escalate the request to the team lead, respond within 10 business days, and the customer must provide a bank statement.
- `f-X-925`: Exception X-925 (amends Policy P-8344). When the customer holds gold status, Policy P-8344 does not apply as written: the agent must instead deny the request, respond within 3 business days, and the customer must provide a government photo ID.

### F7-100-rel-dev-s1000-t032  [ ] OK  [ ] ISSUE

**Prompt:** Case C-742986: customer CU-11502 has submitted a tax-exemption request for $63. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "deny", "approver": "none", "deadline_days": 2, "document": "bank_statement"}`

**Setup:** `{"customers": {"CU-11502": {"region": "UK", "tier": "gold"}}}`

**Gold facts:**
- `f-P-6814`: Policy P-6814 (Tax Exemptions, UK region). For a tax-exemption request from a customer in the UK region with an amount of at least $50 and under $100, the agent must approve the request, respond within 10 business days, and the customer must provide a bank statement.
- `f-X-256`: Exception X-256 (amends Policy P-6814). When the customer holds gold status, Policy P-6814 does not apply as written: the agent must instead deny the request, respond within 2 business days, and the customer must provide a bank statement.

### F7-100-rel-dev-s1000-t031  [ ] OK  [ ] ISSUE

**Prompt:** Case C-645964: customer CU-72332 has submitted a damaged-goods request for $57. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "approve", "approver": "none", "deadline_days": 2, "document": "bank_statement"}`

**Setup:** `{"customers": {"CU-72332": {"region": "LATAM", "tier": "platinum"}}}`

**Gold facts:**
- `f-P-7405`: Policy P-7405 (Damaged Goods, LATAM region). For a damaged-goods request from a customer in the LATAM region with an amount of at least $50 and under $100, the agent must approve the request, respond within 2 business days, and the customer must provide a bank statement.

### F7-100-rel-dev-s1000-t025  [ ] OK  [ ] ISSUE

**Prompt:** Case C-256463: customer CU-39915 has submitted a address-change request for $96. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "deny", "approver": "none", "deadline_days": 9, "document": "bank_statement"}`

**Setup:** `{"customers": {"CU-39915": {"region": "UK", "tier": "platinum"}}}`

**Gold facts:**
- `f-P-7064`: Policy P-7064 (Address Changes, UK region). For a address-change request from a customer in the UK region with an amount of at least $50 and under $100, the agent must deny the request, respond within 9 business days, and the customer must provide a bank statement.
- `f-X-390`: Exception X-390 (amends Policy P-7064). When the customer holds gold status, Policy P-7064 does not apply as written: the agent must instead approve the request, respond within 10 business days, and the customer must provide a government photo ID.

## F7-1000 (F7-1000-rel-dev-s1000)

### F7-1000-rel-dev-s1000-t058  [ ] OK  [ ] ISSUE

**Prompt:** Case C-528350: customer CU-25394 has submitted a warranty request for $11,708. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "escalate", "approver": "compliance_officer", "deadline_days": 9, "document": "receipt"}`

**Setup:** `{"customers": {"CU-25394": {"region": "UK", "tier": "basic"}}}`

**Gold facts:**
- `f-P-9406`: Policy P-9406 (Warranty Claims, UK region). For a warranty request from a customer in the UK region with an amount of at least $10,000 and under $25,000, the agent must escalate the request to the compliance officer, respond within 9 business days, and the customer must provide a receipt.
- `f-X-982`: Exception X-982 (amends Policy P-9406). When the customer holds platinum status, Policy P-9406 does not apply as written: the agent must instead deny the request, respond within 9 business days, and the customer must provide a bank statement.

### F7-1000-rel-dev-s1000-t050  [ ] OK  [ ] ISSUE

**Prompt:** Case C-461371: customer CU-85288 has submitted a billing-dispute request for $22,542. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "approve", "approver": "none", "deadline_days": 6, "document": "police_report"}`

**Setup:** `{"customers": {"CU-85288": {"region": "LATAM", "tier": "silver"}}}`

**Gold facts:**
- `f-P-7285`: Policy P-7285 (Billing Disputes, LATAM region). For a billing-dispute request from a customer in the LATAM region with an amount of at least $10,000 and under $25,000, the agent must approve the request, respond within 6 business days, and the customer must provide a police report.

### F7-1000-rel-dev-s1000-t053  [ ] OK  [ ] ISSUE

**Prompt:** Case C-216913: customer CU-59291 has submitted a damaged-goods request for $10,576. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "escalate", "approver": "tier2_supervisor", "deadline_days": 4, "document": "delivery_photo"}`

**Setup:** `{"customers": {"CU-59291": {"region": "NA", "tier": "gold"}}}`

**Gold facts:**
- `f-P-5797`: Policy P-5797 (Damaged Goods, NA region). For a damaged-goods request from a customer in the NA region with an amount of at least $10,000 and under $25,000, the agent must escalate the request to the Tier-2 supervisor, respond within 4 business days, and the customer must provide a photo of the delivered package.

### F7-1000-rel-dev-s1000-t019  [ ] OK  [ ] ISSUE

**Prompt:** Case C-335933: customer CU-71094 has submitted a address-change request for $166. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "approve", "approver": "none", "deadline_days": 2, "document": "photo_id"}`

**Setup:** `{"customers": {"CU-71094": {"region": "NA", "tier": "silver"}}}`

**Gold facts:**
- `f-P-3123`: Policy P-3123 (Address Changes, NA region). For a address-change request from a customer in the NA region with an amount of at least $100 and under $250, the agent must approve the request, respond within 2 business days, and the customer must provide a government photo ID.

### F7-1000-rel-dev-s1000-t030  [ ] OK  [ ] ISSUE

**Prompt:** Case C-130399: customer CU-29507 has submitted a address-change request for $19,431. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "escalate", "approver": "compliance_officer", "deadline_days": 1, "document": "bank_statement"}`

**Setup:** `{"customers": {"CU-29507": {"region": "EU", "tier": "gold"}}}`

**Gold facts:**
- `f-P-2138`: Policy P-2138 (Address Changes, EU region). For a address-change request from a customer in the EU region with an amount of at least $10,000 and under $25,000, the agent must approve the request, respond within 3 business days, and the customer must provide a receipt.
- `f-X-665`: Exception X-665 (amends Policy P-2138). When the customer holds gold status, Policy P-2138 does not apply as written: the agent must instead escalate the request to the compliance officer, respond within 1 business day, and the customer must provide a bank statement.

### F7-1000-rel-dev-s1000-t022  [ ] OK  [ ] ISSUE

**Prompt:** Case C-740318: customer CU-22790 has submitted a address-change request for $272. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "escalate", "approver": "compliance_officer", "deadline_days": 6, "document": "delivery_photo"}`

**Setup:** `{"customers": {"CU-22790": {"region": "EU", "tier": "basic"}}}`

**Gold facts:**
- `f-P-6949`: Policy P-6949 (Address Changes, EU region). For a address-change request from a customer in the EU region with an amount of at least $250 and under $500, the agent must escalate the request to the compliance officer, respond within 6 business days, and the customer must provide a photo of the delivered package.
- `f-X-654`: Exception X-654 (amends Policy P-6949). When the customer holds platinum status, Policy P-6949 does not apply as written: the agent must instead escalate the request to the compliance officer, respond within 1 business day, and the customer must provide a receipt.

### F7-1000-rel-dev-s1000-t037  [ ] OK  [ ] ISSUE

**Prompt:** Case C-531744: customer CU-28114 has submitted a data-deletion request for $1,256. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "approve", "approver": "none", "deadline_days": 3, "document": "receipt"}`

**Setup:** `{"customers": {"CU-28114": {"region": "APAC", "tier": "gold"}}}`

**Gold facts:**
- `f-P-5410`: Policy P-5410 (Data Deletion, APAC region). For a data-deletion request from a customer in the APAC region with an amount of at least $1,000 and under $2,500, the agent must approve the request, respond within 3 business days, and the customer must provide a receipt.

### F7-1000-rel-dev-s1000-t013  [ ] OK  [ ] ISSUE

**Prompt:** Case C-776837: customer CU-52600 has submitted a warranty request for $55,700. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "deny", "approver": "none", "deadline_days": 2, "document": "none"}`

**Setup:** `{"customers": {"CU-52600": {"region": "EU", "tier": "gold"}}}`

**Gold facts:**
- `f-P-3838`: Policy P-3838 (Warranty Claims, EU region). For a warranty request from a customer in the EU region with an amount of at least $25,000 and under $100,000, the agent must deny the request, respond within 2 business days, and no supporting document is required.

### F7-1000-rel-dev-s1000-t032  [ ] OK  [ ] ISSUE

**Prompt:** Case C-367751: customer CU-87988 has submitted a account-unlock request for $15,176. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "approve", "approver": "none", "deadline_days": 6, "document": "photo_id"}`

**Setup:** `{"customers": {"CU-87988": {"region": "LATAM", "tier": "platinum"}}}`

**Gold facts:**
- `f-P-3990`: Policy P-3990 (Account Lockouts, LATAM region). For a account-unlock request from a customer in the LATAM region with an amount of at least $10,000 and under $25,000, the agent must escalate the request to the regional manager, respond within 3 business days, and no supporting document is required.
- `f-X-773`: Exception X-773 (amends Policy P-3990). When the customer holds platinum status, Policy P-3990 does not apply as written: the agent must instead approve the request, respond within 6 business days, and the customer must provide a government photo ID.

### F7-1000-rel-dev-s1000-t008  [ ] OK  [ ] ISSUE

**Prompt:** Case C-935369: customer CU-90934 has submitted a bulk-order request for $87,713. Look up the customer, determine exactly how company policy requires this request to be handled, and submit your decision with submit_decision.

**Gold:** `{"action": "approve", "approver": "none", "deadline_days": 5, "document": "delivery_photo"}`

**Setup:** `{"customers": {"CU-90934": {"region": "APAC", "tier": "basic"}}}`

**Gold facts:**
- `f-P-5946`: Policy P-5946 (Bulk Orders, APAC region). For a bulk-order request from a customer in the APAC region with an amount of at least $25,000 and under $100,000, the agent must approve the request, respond within 5 business days, and the customer must provide a photo of the delivered package.

## F3-20 (F3-20-dev-s1000)

### F3-20-dev-s1000-t018  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-78709: the customer reports the parcel was lost in transit (order O-842681). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "lost_parcel_escalate", "args": {"order_id": "O-842681", "queue": "Q-902"}}, {"tool": "lost_parcel_reissue", "args": {"order_id": "O-842681", "carrier": "CAR-143"}}]}`

**Setup:** `{"orders": {"O-842681": {"region": "APAC", "status": "open"}}}`

**Gold facts:**
- `f-SOP-lost_parcel-APAC`: Procedure SOP-lost_parcel-APAC. When the customer reports the parcel was lost in transit and the order ships to the APAC region: call lost_parcel_escalate with queue=Q-902; then call lost_parcel_reissue with carrier=CAR-143. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-lost_parcel_escalate`: Tool lost_parcel_escalate (domain: lost parcel). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-lost_parcel_reissue`: Tool lost_parcel_reissue (domain: lost parcel). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-20-dev-s1000-t008  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-89333: the customer reports the parcel was lost in transit (order O-144228). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "lost_parcel_notify", "args": {"order_id": "O-144228", "template": "T-473"}}, {"tool": "lost_parcel_reissue", "args": {"order_id": "O-144228", "carrier": "CAR-874"}}]}`

**Setup:** `{"orders": {"O-144228": {"region": "LATAM", "status": "open"}}}`

**Gold facts:**
- `f-SOP-lost_parcel-LATAM`: Procedure SOP-lost_parcel-LATAM. When the customer reports the parcel was lost in transit and the order ships to the LATAM region: call lost_parcel_notify with template=T-473; then call lost_parcel_reissue with carrier=CAR-874. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-lost_parcel_notify`: Tool lost_parcel_notify (domain: lost parcel). Send the customer a notification using a message template. Parameters: order_id (the order ID) and template (a template code).
- `f-tool-lost_parcel_reissue`: Tool lost_parcel_reissue (domain: lost parcel). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-20-dev-s1000-t048  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-27242: loyalty points were not credited for the order (order O-985556). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "loyalty_adjustment_escalate", "args": {"order_id": "O-985556", "queue": "Q-573"}}, {"tool": "loyalty_adjustment_reissue", "args": {"order_id": "O-985556", "carrier": "CAR-853"}}]}`

**Setup:** `{"orders": {"O-985556": {"region": "NA", "status": "open"}}}`

**Gold facts:**
- `f-SOP-loyalty_adjustment-NA`: Procedure SOP-loyalty_adjustment-NA. When loyalty points were not credited for the order and the order ships to the NA region: call loyalty_adjustment_escalate with queue=Q-573; then call loyalty_adjustment_reissue with carrier=CAR-853. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-loyalty_adjustment_escalate`: Tool loyalty_adjustment_escalate (domain: loyalty adjustment). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-loyalty_adjustment_reissue`: Tool loyalty_adjustment_reissue (domain: loyalty adjustment). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-20-dev-s1000-t006  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-99936: loyalty points were not credited for the order (order O-798588). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "loyalty_adjustment_notify", "args": {"order_id": "O-798588", "template": "T-751"}}, {"tool": "loyalty_adjustment_reissue", "args": {"order_id": "O-798588", "carrier": "CAR-763"}}]}`

**Setup:** `{"orders": {"O-798588": {"region": "UK", "status": "open"}}}`

**Gold facts:**
- `f-SOP-loyalty_adjustment-UK`: Procedure SOP-loyalty_adjustment-UK. When loyalty points were not credited for the order and the order ships to the UK region: call loyalty_adjustment_notify with template=T-751; then call loyalty_adjustment_reissue with carrier=CAR-763. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-loyalty_adjustment_notify`: Tool loyalty_adjustment_notify (domain: loyalty adjustment). Send the customer a notification using a message template. Parameters: order_id (the order ID) and template (a template code).
- `f-tool-loyalty_adjustment_reissue`: Tool loyalty_adjustment_reissue (domain: loyalty adjustment). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-20-dev-s1000-t039  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-72562: the customer reports the parcel was lost in transit (order O-915838). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "lost_parcel_escalate", "args": {"order_id": "O-915838", "queue": "Q-473"}}, {"tool": "lost_parcel_reissue", "args": {"order_id": "O-915838", "carrier": "CAR-800"}}]}`

**Setup:** `{"orders": {"O-915838": {"region": "NA", "status": "open"}}}`

**Gold facts:**
- `f-SOP-lost_parcel-NA`: Procedure SOP-lost_parcel-NA. When the customer reports the parcel was lost in transit and the order ships to the NA region: call lost_parcel_escalate with queue=Q-473; then call lost_parcel_reissue with carrier=CAR-800. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-lost_parcel_escalate`: Tool lost_parcel_escalate (domain: lost parcel). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-lost_parcel_reissue`: Tool lost_parcel_reissue (domain: lost parcel). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-20-dev-s1000-t051  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-54424: loyalty points were not credited for the order (order O-504002). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "loyalty_adjustment_notify", "args": {"order_id": "O-504002", "template": "T-126"}}, {"tool": "loyalty_adjustment_reissue", "args": {"order_id": "O-504002", "carrier": "CAR-307"}}]}`

**Setup:** `{"orders": {"O-504002": {"region": "APAC", "status": "open"}}}`

**Gold facts:**
- `f-SOP-loyalty_adjustment-APAC`: Procedure SOP-loyalty_adjustment-APAC. When loyalty points were not credited for the order and the order ships to the APAC region: call loyalty_adjustment_notify with template=T-126; then call loyalty_adjustment_reissue with carrier=CAR-307. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-loyalty_adjustment_notify`: Tool loyalty_adjustment_notify (domain: loyalty adjustment). Send the customer a notification using a message template. Parameters: order_id (the order ID) and template (a template code).
- `f-tool-loyalty_adjustment_reissue`: Tool loyalty_adjustment_reissue (domain: loyalty adjustment). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-20-dev-s1000-t016  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-95794: loyalty points were not credited for the order (order O-382050). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "loyalty_adjustment_escalate", "args": {"order_id": "O-382050", "queue": "Q-837"}}, {"tool": "loyalty_adjustment_reissue", "args": {"order_id": "O-382050", "carrier": "CAR-956"}}]}`

**Setup:** `{"orders": {"O-382050": {"region": "LATAM", "status": "open"}}}`

**Gold facts:**
- `f-SOP-loyalty_adjustment-LATAM`: Procedure SOP-loyalty_adjustment-LATAM. When loyalty points were not credited for the order and the order ships to the LATAM region: call loyalty_adjustment_escalate with queue=Q-837; then call loyalty_adjustment_reissue with carrier=CAR-956. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-loyalty_adjustment_escalate`: Tool loyalty_adjustment_escalate (domain: loyalty adjustment). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-loyalty_adjustment_reissue`: Tool loyalty_adjustment_reissue (domain: loyalty adjustment). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-20-dev-s1000-t034  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-42619: the customer reports the parcel was lost in transit (order O-121310). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "lost_parcel_notify", "args": {"order_id": "O-121310", "template": "T-942"}}, {"tool": "lost_parcel_reissue", "args": {"order_id": "O-121310", "carrier": "CAR-452"}}]}`

**Setup:** `{"orders": {"O-121310": {"region": "EU", "status": "open"}}}`

**Gold facts:**
- `f-SOP-lost_parcel-EU`: Procedure SOP-lost_parcel-EU. When the customer reports the parcel was lost in transit and the order ships to the EU region: call lost_parcel_notify with template=T-942; then call lost_parcel_reissue with carrier=CAR-452. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-lost_parcel_notify`: Tool lost_parcel_notify (domain: lost parcel). Send the customer a notification using a message template. Parameters: order_id (the order ID) and template (a template code).
- `f-tool-lost_parcel_reissue`: Tool lost_parcel_reissue (domain: lost parcel). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-20-dev-s1000-t045  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-30972: loyalty points were not credited for the order (order O-214851). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "loyalty_adjustment_notify", "args": {"order_id": "O-214851", "template": "T-230"}}, {"tool": "loyalty_adjustment_reissue", "args": {"order_id": "O-214851", "carrier": "CAR-374"}}]}`

**Setup:** `{"orders": {"O-214851": {"region": "EU", "status": "open"}}}`

**Gold facts:**
- `f-SOP-loyalty_adjustment-EU`: Procedure SOP-loyalty_adjustment-EU. When loyalty points were not credited for the order and the order ships to the EU region: call loyalty_adjustment_notify with template=T-230; then call loyalty_adjustment_reissue with carrier=CAR-374. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-loyalty_adjustment_notify`: Tool loyalty_adjustment_notify (domain: loyalty adjustment). Send the customer a notification using a message template. Parameters: order_id (the order ID) and template (a template code).
- `f-tool-loyalty_adjustment_reissue`: Tool loyalty_adjustment_reissue (domain: loyalty adjustment). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-20-dev-s1000-t038  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-75429: the customer reports the parcel was lost in transit (order O-344831). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "lost_parcel_escalate", "args": {"order_id": "O-344831", "queue": "Q-473"}}, {"tool": "lost_parcel_reissue", "args": {"order_id": "O-344831", "carrier": "CAR-800"}}]}`

**Setup:** `{"orders": {"O-344831": {"region": "NA", "status": "open"}}}`

**Gold facts:**
- `f-SOP-lost_parcel-NA`: Procedure SOP-lost_parcel-NA. When the customer reports the parcel was lost in transit and the order ships to the NA region: call lost_parcel_escalate with queue=Q-473; then call lost_parcel_reissue with carrier=CAR-800. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-lost_parcel_escalate`: Tool lost_parcel_escalate (domain: lost parcel). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-lost_parcel_reissue`: Tool lost_parcel_reissue (domain: lost parcel). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

## F3-60 (F3-60-dev-s1000)

### F3-60-dev-s1000-t057  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-64508: the customer reports the parcel was lost in transit (order O-555645). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "lost_parcel_escalate", "args": {"order_id": "O-555645", "queue": "Q-211"}}, {"tool": "lost_parcel_notify", "args": {"order_id": "O-555645", "template": "T-126"}}]}`

**Setup:** `{"orders": {"O-555645": {"region": "APAC", "status": "open"}}}`

**Gold facts:**
- `f-SOP-lost_parcel-APAC`: Procedure SOP-lost_parcel-APAC. When the customer reports the parcel was lost in transit and the order ships to the APAC region: call lost_parcel_escalate with queue=Q-211; then call lost_parcel_notify with template=T-126. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-lost_parcel_escalate`: Tool lost_parcel_escalate (domain: lost parcel). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-lost_parcel_notify`: Tool lost_parcel_notify (domain: lost parcel). Send the customer a notification using a message template. Parameters: order_id (the order ID) and template (a template code).

### F3-60-dev-s1000-t009  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-63915: an item is missing from the package (order O-586858). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "missing_item_escalate", "args": {"order_id": "O-586858", "queue": "Q-228"}}, {"tool": "missing_item_reissue", "args": {"order_id": "O-586858", "carrier": "CAR-780"}}]}`

**Setup:** `{"orders": {"O-586858": {"region": "LATAM", "status": "open"}}}`

**Gold facts:**
- `f-SOP-missing_item-LATAM`: Procedure SOP-missing_item-LATAM. When an item is missing from the package and the order ships to the LATAM region: call missing_item_escalate with queue=Q-228; then call missing_item_reissue with carrier=CAR-780. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-missing_item_escalate`: Tool missing_item_escalate (domain: missing item). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-missing_item_reissue`: Tool missing_item_reissue (domain: missing item). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-60-dev-s1000-t019  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-12614: an item is missing from the package (order O-397912). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "missing_item_escalate", "args": {"order_id": "O-397912", "queue": "Q-228"}}, {"tool": "missing_item_reissue", "args": {"order_id": "O-397912", "carrier": "CAR-780"}}]}`

**Setup:** `{"orders": {"O-397912": {"region": "LATAM", "status": "open"}}}`

**Gold facts:**
- `f-SOP-missing_item-LATAM`: Procedure SOP-missing_item-LATAM. When an item is missing from the package and the order ships to the LATAM region: call missing_item_escalate with queue=Q-228; then call missing_item_reissue with carrier=CAR-780. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-missing_item_escalate`: Tool missing_item_escalate (domain: missing item). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-missing_item_reissue`: Tool missing_item_reissue (domain: missing item). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-60-dev-s1000-t006  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-11284: the order was placed under a duplicate account (order O-235171). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "account_merge_hold", "args": {"order_id": "O-235171", "hold_reason": "HR-453"}}, {"tool": "account_merge_reroute", "args": {"order_id": "O-235171", "depot_code": "DEP-640"}}]}`

**Setup:** `{"orders": {"O-235171": {"region": "EU", "status": "open"}}}`

**Gold facts:**
- `f-SOP-account_merge-EU`: Procedure SOP-account_merge-EU. When the order was placed under a duplicate account and the order ships to the EU region: call account_merge_hold with hold_reason=HR-453; then call account_merge_reroute with depot_code=DEP-640. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-account_merge_hold`: Tool account_merge_hold (domain: account merge). Place an order on hold with a hold-reason code. Parameters: order_id (the order ID) and hold_reason (a hold reason code).
- `f-tool-account_merge_reroute`: Tool account_merge_reroute (domain: account merge). Reroute an in-transit order through a depot. Parameters: order_id (the order ID) and depot_code (a depot code code).

### F3-60-dev-s1000-t046  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-84561: the customer reports the parcel was lost in transit (order O-616717). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "lost_parcel_escalate", "args": {"order_id": "O-616717", "queue": "Q-211"}}, {"tool": "lost_parcel_notify", "args": {"order_id": "O-616717", "template": "T-126"}}]}`

**Setup:** `{"orders": {"O-616717": {"region": "APAC", "status": "open"}}}`

**Gold facts:**
- `f-SOP-lost_parcel-APAC`: Procedure SOP-lost_parcel-APAC. When the customer reports the parcel was lost in transit and the order ships to the APAC region: call lost_parcel_escalate with queue=Q-211; then call lost_parcel_notify with template=T-126. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-lost_parcel_escalate`: Tool lost_parcel_escalate (domain: lost parcel). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-lost_parcel_notify`: Tool lost_parcel_notify (domain: lost parcel). Send the customer a notification using a message template. Parameters: order_id (the order ID) and template (a template code).

### F3-60-dev-s1000-t004  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-34319: the order was placed under a duplicate account (order O-640856). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "account_merge_credit", "args": {"order_id": "O-640856", "credit_code": "CR-983"}}, {"tool": "account_merge_reroute", "args": {"order_id": "O-640856", "depot_code": "DEP-810"}}]}`

**Setup:** `{"orders": {"O-640856": {"region": "APAC", "status": "open"}}}`

**Gold facts:**
- `f-SOP-account_merge-APAC`: Procedure SOP-account_merge-APAC. When the order was placed under a duplicate account and the order ships to the APAC region: call account_merge_credit with credit_code=CR-983; then call account_merge_reroute with depot_code=DEP-810. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-account_merge_credit`: Tool account_merge_credit (domain: account merge). Apply a goodwill credit to the customer's account for the order. Parameters: order_id (the order ID) and credit_code (a credit code code).
- `f-tool-account_merge_reroute`: Tool account_merge_reroute (domain: account merge). Reroute an in-transit order through a depot. Parameters: order_id (the order ID) and depot_code (a depot code code).

### F3-60-dev-s1000-t043  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-46811: an item is missing from the package (order O-751774). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "missing_item_notify", "args": {"order_id": "O-751774", "template": "T-354"}}, {"tool": "missing_item_reissue", "args": {"order_id": "O-751774", "carrier": "CAR-918"}}]}`

**Setup:** `{"orders": {"O-751774": {"region": "APAC", "status": "open"}}}`

**Gold facts:**
- `f-SOP-missing_item-APAC`: Procedure SOP-missing_item-APAC. When an item is missing from the package and the order ships to the APAC region: call missing_item_notify with template=T-354; then call missing_item_reissue with carrier=CAR-918. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-missing_item_notify`: Tool missing_item_notify (domain: missing item). Send the customer a notification using a message template. Parameters: order_id (the order ID) and template (a template code).
- `f-tool-missing_item_reissue`: Tool missing_item_reissue (domain: missing item). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-60-dev-s1000-t021  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-11325: the order was placed under a duplicate account (order O-613549). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "account_merge_hold", "args": {"order_id": "O-613549", "hold_reason": "HR-453"}}, {"tool": "account_merge_reroute", "args": {"order_id": "O-613549", "depot_code": "DEP-640"}}]}`

**Setup:** `{"orders": {"O-613549": {"region": "EU", "status": "open"}}}`

**Gold facts:**
- `f-SOP-account_merge-EU`: Procedure SOP-account_merge-EU. When the order was placed under a duplicate account and the order ships to the EU region: call account_merge_hold with hold_reason=HR-453; then call account_merge_reroute with depot_code=DEP-640. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-account_merge_hold`: Tool account_merge_hold (domain: account merge). Place an order on hold with a hold-reason code. Parameters: order_id (the order ID) and hold_reason (a hold reason code).
- `f-tool-account_merge_reroute`: Tool account_merge_reroute (domain: account merge). Reroute an in-transit order through a depot. Parameters: order_id (the order ID) and depot_code (a depot code code).

### F3-60-dev-s1000-t030  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-52964: an item is missing from the package (order O-545749). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "missing_item_escalate", "args": {"order_id": "O-545749", "queue": "Q-150"}}, {"tool": "missing_item_reissue", "args": {"order_id": "O-545749", "carrier": "CAR-184"}}]}`

**Setup:** `{"orders": {"O-545749": {"region": "UK", "status": "open"}}}`

**Gold facts:**
- `f-SOP-missing_item-UK`: Procedure SOP-missing_item-UK. When an item is missing from the package and the order ships to the UK region: call missing_item_escalate with queue=Q-150; then call missing_item_reissue with carrier=CAR-184. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-missing_item_escalate`: Tool missing_item_escalate (domain: missing item). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-missing_item_reissue`: Tool missing_item_reissue (domain: missing item). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

### F3-60-dev-s1000-t035  [ ] OK  [ ] ISSUE

**Prompt:** Ticket K-92419: the customer reports the parcel was lost in transit (order O-548580). Resolve it exactly as the standard operating procedure requires, then call finish.

**Gold:** `{"calls": [{"tool": "lost_parcel_escalate", "args": {"order_id": "O-548580", "queue": "Q-260"}}, {"tool": "lost_parcel_reissue", "args": {"order_id": "O-548580", "carrier": "CAR-750"}}]}`

**Setup:** `{"orders": {"O-548580": {"region": "EU", "status": "open"}}}`

**Gold facts:**
- `f-SOP-lost_parcel-EU`: Procedure SOP-lost_parcel-EU. When the customer reports the parcel was lost in transit and the order ships to the EU region: call lost_parcel_escalate with queue=Q-260; then call lost_parcel_reissue with carrier=CAR-750. Pass the order ID as order_id. Do not call any other mutating tool.
- `f-tool-lost_parcel_escalate`: Tool lost_parcel_escalate (domain: lost parcel). Escalate the order case to a specialist queue. Parameters: order_id (the order ID) and queue (a queue code).
- `f-tool-lost_parcel_reissue`: Tool lost_parcel_reissue (domain: lost parcel). Re-ship the items of an order through the given carrier service. Parameters: order_id (the order ID) and carrier (a carrier code).

## F5-1hop (F5-1hop-dev-s1000)

### F5-1hop-dev-s1000-t006  [ ] OK  [ ] ISSUE

**Prompt:** Who was the manager of the Lumen team on 2023-10-25? Submit the full name with submit_answer.

**Gold:** `{"answer": "Kwame Mensah"}`

**Gold facts:**
- `f-E-0066`: Effective 2021-01-01, Kwame Mensah became manager of the Lumen team.

### F5-1hop-dev-s1000-t022  [ ] OK  [ ] ISSUE

**Prompt:** Who was the manager of the Cobalt team on 2024-11-30? Submit the full name with submit_answer.

**Gold:** `{"answer": "Oskar Nowak"}`

**Gold facts:**
- `f-E-0014`: Effective 2024-05-11, Oskar Nowak became manager of the Cobalt team.

### F5-1hop-dev-s1000-t027  [ ] OK  [ ] ISSUE

**Prompt:** Who was the manager of the Iris team on 2025-04-29? Submit the full name with submit_answer.

**Gold:** `{"answer": "Mateo Silva"}`

**Gold facts:**
- `f-E-0050`: Effective 2021-05-07, Mateo Silva became manager of the Iris team.

### F5-1hop-dev-s1000-t020  [ ] OK  [ ] ISSUE

**Prompt:** Who was the manager of the Granite team on 2023-03-27? Submit the full name with submit_answer.

**Gold:** `{"answer": "Jonas Weber"}`

**Gold facts:**
- `f-E-0038`: Effective 2022-09-10, Jonas Weber became manager of the Granite team.

### F5-1hop-dev-s1000-t039  [ ] OK  [ ] ISSUE

**Prompt:** Who was the manager of the Ember team on 2022-02-18? Submit the full name with submit_answer.

**Gold:** `{"answer": "Kenji Sato"}`

**Gold facts:**
- `f-E-0024`: Effective 2021-01-01, Kenji Sato became manager of the Ember team.

### F5-1hop-dev-s1000-t040  [ ] OK  [ ] ISSUE

**Prompt:** Who was the manager of the Ember team on 2025-01-01? Submit the full name with submit_answer.

**Gold:** `{"answer": "Kenji Sato"}`

**Gold facts:**
- `f-E-0024`: Effective 2021-01-01, Kenji Sato became manager of the Ember team.

### F5-1hop-dev-s1000-t013  [ ] OK  [ ] ISSUE

**Prompt:** Who was the manager of the Ember team on 2023-10-11? Submit the full name with submit_answer.

**Gold:** `{"answer": "Kenji Sato"}`

**Gold facts:**
- `f-E-0024`: Effective 2021-01-01, Kenji Sato became manager of the Ember team.

### F5-1hop-dev-s1000-t035  [ ] OK  [ ] ISSUE

**Prompt:** Who was the manager of the Ember team on 2023-01-02? Submit the full name with submit_answer.

**Gold:** `{"answer": "Kenji Sato"}`

**Gold facts:**
- `f-E-0024`: Effective 2021-01-01, Kenji Sato became manager of the Ember team.

### F5-1hop-dev-s1000-t030  [ ] OK  [ ] ISSUE

**Prompt:** Who was the manager of the Atlas team on 2021-02-05? Submit the full name with submit_answer.

**Gold:** `{"answer": "Nina Larsen"}`

**Gold facts:**
- `f-E-0000`: Effective 2021-01-01, Nina Larsen became manager of the Atlas team.

### F5-1hop-dev-s1000-t028  [ ] OK  [ ] ISSUE

**Prompt:** Who was the manager of the Fjord team on 2021-06-21? Submit the full name with submit_answer.

**Gold:** `{"answer": "Amara Okafor"}`

**Gold facts:**
- `f-E-0030`: Effective 2021-01-01, Amara Okafor became manager of the Fjord team.

## F5-2hop (F5-2hop-dev-s1000)

### F5-2hop-dev-s1000-t055  [ ] OK  [ ] ISSUE

**Prompt:** On 2023-02-02, in which city was the team managed by Ben Carter based? Submit the city with submit_answer.

**Gold:** `{"answer": "Toronto"}`

**Gold facts:**
- `f-E-0030`: Effective 2021-01-01, Ben Carter became manager of the Fjord team.
- `f-E-0034`: Effective 2021-02-07, the Fjord team is based in the Toronto office.

### F5-2hop-dev-s1000-t033  [ ] OK  [ ] ISSUE

**Prompt:** On 2025-08-22, in which city was the team managed by Lucia Romero based? Submit the city with submit_answer.

**Gold:** `{"answer": "Santiago"}`

**Gold facts:**
- `f-E-0051`: Effective 2023-03-17, Lucia Romero became manager of the Iris team.
- `f-E-0053`: Effective 2024-06-23, the Iris team is based in the Santiago office.

### F5-2hop-dev-s1000-t016  [ ] OK  [ ] ISSUE

**Prompt:** On 2021-10-09, in which city was the team managed by Diego Ramos based? Submit the city with submit_answer.

**Gold:** `{"answer": "Denver"}`

**Gold facts:**
- `f-E-0012`: Effective 2021-01-01, Diego Ramos became manager of the Cobalt team.
- `f-E-0013`: Effective 2021-01-01, the Cobalt team is based in the Denver office.

### F5-2hop-dev-s1000-t003  [ ] OK  [ ] ISSUE

**Prompt:** On 2025-07-17, in which city was the team managed by Lucia Romero based? Submit the city with submit_answer.

**Gold:** `{"answer": "Santiago"}`

**Gold facts:**
- `f-E-0051`: Effective 2023-03-17, Lucia Romero became manager of the Iris team.
- `f-E-0053`: Effective 2024-06-23, the Iris team is based in the Santiago office.

### F5-2hop-dev-s1000-t051  [ ] OK  [ ] ISSUE

**Prompt:** On 2024-12-25, in which city was the team managed by Ravi Patel based? Submit the city with submit_answer.

**Gold:** `{"answer": "Osaka"}`

**Gold facts:**
- `f-E-0020`: Effective 2024-02-02, Ravi Patel became manager of the Delta team.
- `f-E-0023`: Effective 2024-01-15, the Delta team is based in the Osaka office.

### F5-2hop-dev-s1000-t035  [ ] OK  [ ] ISSUE

**Prompt:** On 2021-05-08, in which city was the team managed by Ben Carter based? Submit the city with submit_answer.

**Gold:** `{"answer": "Toronto"}`

**Gold facts:**
- `f-E-0030`: Effective 2021-01-01, Ben Carter became manager of the Fjord team.
- `f-E-0034`: Effective 2021-02-07, the Fjord team is based in the Toronto office.

### F5-2hop-dev-s1000-t000  [ ] OK  [ ] ISSUE

**Prompt:** On 2023-06-26, in which city was the team managed by Lucia Romero based? Submit the city with submit_answer.

**Gold:** `{"answer": "Melbourne"}`

**Gold facts:**
- `f-E-0051`: Effective 2023-03-17, Lucia Romero became manager of the Iris team.
- `f-E-0052`: Effective 2022-04-24, the Iris team is based in the Melbourne office.

### F5-2hop-dev-s1000-t005  [ ] OK  [ ] ISSUE

**Prompt:** On 2025-10-27, in which city was the team managed by Omar Li based? Submit the city with submit_answer.

**Gold:** `{"answer": "Osaka"}`

**Gold facts:**
- `f-E-0021`: Effective 2025-05-25, Omar Li became manager of the Delta team.
- `f-E-0023`: Effective 2024-01-15, the Delta team is based in the Osaka office.

### F5-2hop-dev-s1000-t046  [ ] OK  [ ] ISSUE

**Prompt:** On 2021-09-05, in which city was the team managed by Ben Carter based? Submit the city with submit_answer.

**Gold:** `{"answer": "Toronto"}`

**Gold facts:**
- `f-E-0030`: Effective 2021-01-01, Ben Carter became manager of the Fjord team.
- `f-E-0034`: Effective 2021-02-07, the Fjord team is based in the Toronto office.

### F5-2hop-dev-s1000-t025  [ ] OK  [ ] ISSUE

**Prompt:** On 2022-03-16, in which city was the team managed by Aiko Tanaka based? Submit the city with submit_answer.

**Gold:** `{"answer": "Denver"}`

**Gold facts:**
- `f-E-0042`: Effective 2021-01-01, Aiko Tanaka became manager of the Harbor team.
- `f-E-0043`: Effective 2021-01-01, the Harbor team is based in the Denver office.
