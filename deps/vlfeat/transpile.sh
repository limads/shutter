git clone https://github.com/vlfeat/vlfeat.git
cd vlfeat-master
mkdir vl-rs
c2rust transpile compile_commands.json -e -o vl-rs 

# TODO replace with (when ready)
# c2rust refactor rewrite_expr 'vl_log2_d' 'f64::log2'
find src -type f -exec sed -i 's/vl_log2_d/f64::log2/g' {} \;
find src -type f -exec sed -i 's/vl_log2_f/f32::log2/g' {} \;
find src -type f -exec sed -i 's/vl_sqrt_d/f64::sqrt/g' {} \;
find src -type f -exec sed -i 's/vl_sqrt_f/f32::sqrt/g' {} \;
find src -type f -exec sed -i 's/vl_round_d/f64::round/g' {} \;
find src -type f -exec sed -i 's/vl_round_f/f32::round/g' {} \;
find src -type f -exec sed -i 's/vl_ceil_d/f64::ceil/g' {} \;
find src -type f -exec sed -i 's/vl_ceil_f/f32::ceil/g' {} \;

# Manual fixes:
# Add 'as libc::c_long' at svm.rs:1778
# Add 'as libc::c_long' at svm.rs:1782
# Comment #[feature(...)] macros and modules at lib.rs: svm svmdataset pgm, generic,
# getopt_long, covdet since they require unstable Rust features (using 1.65 stable).
