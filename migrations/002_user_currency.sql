-- Add currency preference and country to users table
-- Currency is auto-detected from location on sign-up, editable on profile page

ALTER TABLE users ADD COLUMN currency CHAR(3) DEFAULT 'GBP' AFTER picture;
ALTER TABLE users ADD COLUMN country VARCHAR(100) DEFAULT NULL AFTER currency;
