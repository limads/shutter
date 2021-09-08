/// Iterates over pairs of pixels within a column, carrying row index of the top element at first position.
pub fn vertical_col_iterator<'a, N>(
    rows : impl Iterator<Item=&'a [N]>+Clone+ 'a,
    comp_dist : usize,
    col : usize
) -> impl Iterator<Item=(usize, (&'a N, &'a N))> + 'a
where
    N : Copy + 'a
{
    let n = rows.clone().count();
    rows.clone().take(n - comp_dist)
        .map(move |row| &row[col] )
        .zip(rows.skip(comp_dist).map(move |row| &row[col] ))
        .enumerate()
}

/// Iterates over pairs of pixels within a row, carrying the column index of the left element at first position
pub fn horizontal_row_iterator<'a, N>(
    row : &'a [N],
    comp_dist : usize
) -> impl Iterator<Item=(usize, (&'a N, &'a N))>+'a {
    row[0..(row.len().saturating_sub(comp_dist))].iter()
        .zip(row[comp_dist..row.len()].iter())
        .enumerate()
}

/// Iterates over pairs of pixels, carrying row index at first position for the informed column.
pub fn vertical_row_iterator<'a, N>(
    rows : impl Iterator<Item=&'a [N]> + Clone + 'a,
    comp_dist : usize,
    col : usize
) -> impl Iterator<Item=(usize, (&'a N, &'a N))>+'a
where
    N : Copy + 'a
{
    let lower_rows = rows.clone().step_by(comp_dist);
    let upper_rows = rows.skip(comp_dist).step_by(comp_dist);
    lower_rows.zip(upper_rows)
        .enumerate()
        .map(move |(row_ix, (lower, upper))| (comp_dist*row_ix, (&lower[col], &upper[col])) )
}

