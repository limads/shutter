```toml
shutter = { path = "/home/diego/Software/shutter", features=["literate"] }
```

```rust
use shutter::image::*;
use shutter::segmentation::*;

let mut img = Image::new_constant(240, 240, 0);
/*for r in 80..160 {
    for c in 80..160 {  	
        img[(r, c)] = 255;
    }
}*/

for r in 0..240 {
	for c in 0..240 {
		let dist = ((r as f64 - 120 as f64).powf(2.) + 
			(c as f64 - 120 as f64).powf(2.)).sqrt();
		if dist < 60. {
			img[(r, c)] = 255;
		}
	}
}

let patch = Patch::grow(&img.full_window(), (110, 110), 1, ColorMode::Exact(255), ReferenceMode::Constant, None, ExpansionMode::Contour).unwrap();

let mut cont_img = Image::new_constant(240, 240, 0);
cont_img.draw(Mark::Shape(patch.pxs.clone(), 255));
let rect = patch.outer_rect();
img.draw(Mark::Rect((rect.0 - 1, rect.1 - 1), (rect.2, rect.3), 127));

let mut pts_img = Image::new_constant(240, 240, 0);
for px in patch.pxs.iter() {
	pts_img.draw(Mark::Cross(*px, 4, 255));	
}

(img, cont_img, pts_img)
```

