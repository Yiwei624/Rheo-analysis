CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    sample_name TEXT,
    geometry_type TEXT,
    operator TEXT,
    notes TEXT,
    source_filename TEXT,
    r1_m REAL,
    r2_m REAL,
    height_m REAL,
    yield0_pa REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    row_index INTEGER NOT NULL,
    segment TEXT,
    shear_rate_1_s REAL,
    shear_stress_pa REAL,
    viscosity_pa_s REAL,
    target_shear_rate_1_s REAL,
    percentage_deviation_pct REAL,
    temperature_c REAL,
    time_s REAL,
    thrust_g REAL,
    accumulated_time_s REAL,
    torque_nm REAL,
    angular_velocity_rad_s REAL,
    notes TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE INDEX IF NOT EXISTS idx_data_points_experiment_id ON data_points(experiment_id);
