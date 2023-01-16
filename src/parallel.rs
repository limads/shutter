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

/* A RAII guard for data that is split and sent to multiple threads. The original
value is blocked from being accessed until the destructor of all Part guards
is called. It is up to the implementor to guarantee the safety of the split operation.
The split cannot yield overlapping memory regions. The split is mostly useful with
mutable memory regions, that can be moved to the split call without aliasing guaranteed
by the compiler.
pub struct Partition<T, P> {
    counter : AtomicU32,
    total : T,
    pub parts : Vec<Part<T>>
}

impl<T, P> Partition<T, P> {

    // Returns total when all part guards have been died.
    pub fn wait_total(self) -> T {
        loop {
            if self.counter() == 0 {
                return self.total;
            }
            thread::yield()
        }
    }

}

pub trait Split {

    unsafe fn split(total : Self, parts : usize) -> Partition<Self>;

}

impl Split for ImagePtr<P> {

}

unsafe impl Send for Part<ImagePtr<P>>;

pub struct Part<T> {
    counter : AtomicU32
    val : T
}

impl AsRef<T> for Part<T> { }

impl AsMut<T> for Part<T> { }

impl Drop for Part<T> {

    fn drop(&mut self) {
        self.counter -= 1;
    }
}

