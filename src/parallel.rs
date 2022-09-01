/*pub trait Offset 
where
    Self : Clone
{

    fn offset_mut(&mut self);
    
    fn offset(&self) -> Self {
        let other = self.clone();
        other.offset_mut();
        other
    }
    
}

pub trait Algorithm 
where
    Self : Default
{

    pub fn parallelize(self, threads : usize) -> Parallel<Self> {
        Parallel { alg : self, threads }
    }
    
    // type Input;
    
    // Type Output : Offset;
    
    // pub fn update(&mut self, input : Self::Input);
    
    // pub fn calculate<I>(input : Self::Input) {
    //     let mut alg = Self::default();
    //     alg.update(input)
    // }
    
}

/*
A parallel algorithm has a "split" step, where the input image is split into 4
sub-images and the algorithm state is clonsed into scoped threads, and a join stage
where the results of each thread are offset by the position of the original image.
*/
pub struct Parallel<A> 
    where A : Algorithm
{

    alg : A,
    
    threads : usize
    
}

impl<A> Parallel<A>
    where A : Algorithm
{

    // pub fn calculate_parallel(input : A::Input) -> Self
    
    // pub fn update_parallel(&mut self, A::Input) {
        // thread::scope
        // for i in 0..self.threads {
        // scope.spawn(|t|
        //
        //
        // 
        // t.join()
        // t.offset_mut(tl);
    // }
    
}*/

