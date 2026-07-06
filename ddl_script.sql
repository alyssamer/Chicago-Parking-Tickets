--------------------------------
--- main tickets table --- 
CREATE TABLE tickets (
    ticket_number VARCHAR,
    issue_date TIMESTAMP,
    violation_location VARCHAR,
    license_plate_number VARCHAR,
    license_plate_state VARCHAR,
    license_plate_type VARCHAR,
    zipcode VARCHAR,
    violation_code VARCHAR,
    violation_description VARCHAR,
    unit VARCHAR,
    unit_description VARCHAR,
    vehicle_make VARCHAR,
    fine_level1_amount NUMERIC,
    fine_level2_amount NUMERIC,
    current_amount_due NUMERIC,
    total_payments NUMERIC,
    ticket_queue VARCHAR,
    ticket_queue_date TIMESTAMP,
    notice_level VARCHAR,
    notice_number VARCHAR,
    hearing_disposition VARCHAR,
    officer VARCHAR,
    normalized_address VARCHAR,
    year INT,
    month INT,
    hour INT,
    ward VARCHAR,
    tract_id VARCHAR,
    blockgroup_geoid VARCHAR,
    community_area_number VARCHAR,
    community_area_name VARCHAR,
    geocode_accuracy NUMERIC,
    geocode_accuracy_type VARCHAR,
    geocoded_address VARCHAR,
    geocoded_lng NUMERIC,
    geocoded_lat NUMERIC
);
	

--- target variable --- 
ALTER TABLE tickets ADD COLUMN target VARCHAR;

UPDATE tickets
SET target = CASE
    WHEN ticket_queue = 'Paid' THEN 'Paid'
    WHEN ticket_queue = 'Dismissed' THEN 'Dismissed'
    ELSE 'Unpaid';


--- cleaning ! ---
--- null or non-numeric zips ---
UPDATE tickets 
SET zipcode = NULL
WHERE zipcode = '000000000' OR zipcode !~ '[0-9]';


--- repeat offenders ---
SELECT license_plate_number, COUNT(*) as ticket_count
FROM tickets
GROUP BY license_plate_number
ORDER BY ticket_count DESC
LIMIT 20;



--------------------------------
--- sample for modeling ---
--- 7% ---
CREATE TABLE tickets_sample AS
SELECT *
FROM tickets
WHERE year BETWEEN 2000 AND 2018
AND ticket_queue != 'Hearing Req'
AND RANDOM() < 0.07;


--------------------------------
--- sample from one full year --- 
CREATE TABLE tickets_2017 AS
SELECT *
FROM tickets
WHERE year = 2017
AND ticket_queue != 'Hearing Req';







