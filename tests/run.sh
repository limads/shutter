cargo test --no-run

for item in $(cat tests/tests.sql | sqlite3) ; do 
    cargo test -- ${item} --nocapture
done


