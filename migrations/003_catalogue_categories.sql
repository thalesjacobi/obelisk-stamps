-- Catalogue: add category and slug for SEO-friendly URLs like /catalogue/<category>/<slug>

ALTER TABLE catalogue
  ADD COLUMN category VARCHAR(100) DEFAULT NULL,
  ADD COLUMN slug VARCHAR(200) DEFAULT NULL;

CREATE INDEX idx_catalogue_category ON catalogue(category);

-- Uniqueness per category (MySQL treats multiple NULLs as distinct, so legacy rows are fine)
CREATE UNIQUE INDEX idx_catalogue_cat_slug ON catalogue(category, slug);
