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

/// Iterates over a diagonal, starting at given (row, col) and going from top-left to bottom-right
pub fn diagonal_right_row_iterator<'a, N>(
    rows : impl Iterator<Item=&'a [N]> + Clone + 'a,
    comp_dist : usize,
    start : (usize, usize)
) -> impl Iterator<Item=((usize, usize), (&'a N, &'a N))>+'a
where
    N : Copy + 'a
{
    let nrows = rows.clone().count();
    let ncols = rows.clone().next().unwrap().len();
    let take_n = (nrows.saturating_sub(start.0) / comp_dist).min(ncols.saturating_sub(start.1) / comp_dist).saturating_sub(1);
    let lower_rows = rows.clone().skip(start.0).step_by(comp_dist);
    let upper_rows = rows.clone().skip(start.0+comp_dist).step_by(comp_dist);
    lower_rows.zip(upper_rows)
        .enumerate()
        .take(take_n)
        .map(move |(ix, (row1, row2))| {
            let px_ix = (start.0 + comp_dist*ix, start.1 + comp_dist*ix);
            // println!("{:?}", (start, nrows, ncols, take_n, px_ix));
            (px_ix, (&row1[px_ix.1], &row2[px_ix.1 + comp_dist]))
         })
}

// Iterates over a diagonal, starting at given (row, col) and going from top-right to bottom-left
pub fn diagonal_left_row_iterator<'a, N>(
    rows : impl Iterator<Item=&'a [N]> + Clone + 'a,
    comp_dist : usize,
    start : (usize, usize)
) -> impl Iterator<Item=((usize, usize), (&'a N, &'a N))>+'a
where
    N : Copy + 'a
{
    let nrows = rows.clone().count();
    let ncols = rows.clone().next().unwrap().len();
    let take_n = (nrows.saturating_sub(start.0) / comp_dist).min(start.1 / comp_dist).saturating_sub(1);
    let lower_rows = rows.clone().skip(start.0).step_by(comp_dist);
    let upper_rows = rows.skip(start.0+comp_dist).step_by(comp_dist);
    lower_rows.zip(upper_rows)
        .enumerate()
        .take(take_n)
        .map(move |(ix, (row1, row2))| {
            let px_ix = (start.0 + comp_dist*ix, start.1 - comp_dist*ix);
            // println!("{:?}", (start, nrows, ncols, take_n, ncols, px_ix));
            (px_ix, (&row1[px_ix.1], &row2[px_ix.1 - comp_dist]))
         })
}

#[test]
fn diag() {
    use crate::image::Image;
    let mut img = Image::new_constant(10, 10, 1);

    for r in 0..10 {
        for c in 0..10 {
            img[(r, c)] = r;
        }
    }

    println!("Diagonal right:");
    diagonal_right_row_iterator(img.full_window().rows(), 2, (0, 0))
        .for_each(|(ix, px)| println!("{:?} : {:?}", ix, px) );

    println!("Diagonal left:");
    diagonal_left_row_iterator(img.full_window().rows(), 2, (0, 9))
        .for_each(|(ix, px)| println!("{:?} : {:?}", ix, px) );
}

