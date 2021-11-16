```shell
bundle update && weave examples/dbscan.md -o examples/dbscan.html
```

```rust
use bayes::prob::*;
use plots::*;
use shutter::feature::point::dbscan::*;

const PTS_PER_CLUSTER : usize = 1000;

let n1 = Normal::prior(100.0, Some(30.0));
let n2 = Normal::prior(100.0, Some(30.0));

let n3 = Normal::prior(200.0, Some(30.0));
let n4 = Normal::prior(200.0, Some(30.0));

let n5 = Normal::prior(150.0, Some(200.0));
let n6 = Normal::prior(150.0, Some(200.0));

let mut rng = rand::thread_rng();
let mut rng2 = rand::thread_rng();
let mut rng3 = rand::thread_rng();
let pts : Vec<(usize, usize)> = (0..PTS_PER_CLUSTER).map(|_| (n1.sample(&mut rng) as usize, n2.sample(&mut rng) as usize) )
    .chain((0..PTS_PER_CLUSTER).map(|_| (n3.sample(&mut rng2) as usize, n4.sample(&mut rng2) as usize)))
    .chain((0..PTS_PER_CLUSTER).map(|_| (n5.sample(&mut rng3) as usize, n6.sample(&mut rng3) as usize)))
    .collect();

let (clusts, noise) = dbscan(&pts, 5.0, 10, true);

let mut pl = Plot::new().scale_x(Scale::new().adjustment(Adjustment::Tight)).scale_y(Scale::new().adjustment(Adjustment::Tight));
let colors = ["#FF0000", "#00FF00", "#0000FF", "#00FFFF", "#FFFF00", "#FFFFF0"];
for (ix, (_, clust_pts)) in clusts.iter().enumerate() {
    assert!(clust_pts.len() >= 1);
    let x = clust_pts.iter().map(|pt| pt.0 as f64 );
    let y = clust_pts.iter().map(|pt| pt.1 as f64 );
    
    pl = pl.draw(ScatterMapping::map(x, y).color(colors[ix].to_string()).radius(3.0)); 
}

let x = noise.iter().map(|pt| pt.0 as f64 );
let y = noise.iter().map(|pt| pt.1 as f64 );
pl = pl.draw(ScatterMapping::map(x, y).color("#000000".to_string()).radius(3.0));

pl
```

```
let clusters_len = clusts.iter().map(|(_, v)| v.len() ).collect::<Vec<_>>();
clusters_len
```

```
let noise_len = noise.len();
noise_len
```

```
use std::time::SystemTime;

let t1 = SystemTime::now();
let _ = dbscan(&pts, 5.0, 10, false);
let t2 = SystemTime::now();
let _ = dbscan(&pts, 5.0, 10, true);
let t3 = SystemTime::now();

let no_indexing = t2.duration_since(t1).unwrap().as_millis();
let with_indexing = t3.duration_since(t2).unwrap().as_millis();
(no_indexing, with_indexing) 
```
