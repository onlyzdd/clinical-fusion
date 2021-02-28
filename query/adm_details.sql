DROP MATERIALIZED VIEW IF EXISTS adm_details CASCADE;
CREATE MATERIALIZED VIEW adm_details as
(
    select p.subject_id, p.gender, p.dob, p.dod, hadm_id, admittime, dischtime, admission_type, insurance, marital_status, ethnicity, hospital_expire_flag, has_chartevents_data
    from admissions adm
    join patients p
    on adm.subject_id = p.subject_id
)
