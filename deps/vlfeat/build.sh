# Final args: -lm

clang -c vl/random.c -Ivl -o build/random.o
clang -c vl/generic.c -Ivl -o build/generic.o
clang -c vl/mathop.c -Ivl -o build/mathop.o
clang -c vl/aib.c -Ivl -o build/aib.o
clang -c vl/array.c -Ivl -o build/array.o
clang -c vl/covdet.c -Ivl -o build/covdet.o
clang -c vl/dsift.c -Ivl -o build/dsift.o
clang -c vl/fisher.c -Ivl -o build/fisher.o
clang -c vl/getopt_long.c -Ivl -o build/getopt_long.o
clang -c vl/gmm.c -Ivl -o build/gmm.o
clang -c vl/hikmeans.c -Ivl -o build/hikmeans.o
clang -c vl/homkermap.c -Ivl -o build/homkermap.o
clang -c vl/host.c -Ivl -o build/host.o
clang -c vl/ikmeans.c -Ivl -o build/ikmeans.o
clang -c vl/imopv.c -Ivl -o build/imopv.o
clang -c vl/kdtree.c -Ivl -o build/kdtree.o
clang -c vl/kmeans.c -Ivl -o build/kmeans.o
clang -c vl/lbp.c -Ivl -o build/lbp.o
clang -c vl/liop.c -Ivl -o build/liop.o
clang -c vl/mser.c -Ivl -o build/mser.o
clang -c vl/pgm.c -Ivl -o build/pgm.o
clang -c vl/quickshift.c -Ivl -o build/quickshift.o
clang -c vl/random.c -Ivl -o build/random.o
clang -c vl/rodrigues.c -Ivl -o build/rodrigues.o
clang -c vl/scalespace.c -Ivl -o build/scalespace.o
clang -c vl/sift.c -Ivl -o build/sift.o
clang -c vl/slic.c -Ivl -o build/slic.o
clang -c vl/stringop.c -Ivl -o build/stringop.o
clang -c vl/svm.c -Ivl -o build/svm.o
clang -c vl/svmdataset.c -Ivl -o build/svmdataset.o
clang -c vl/vlad.c -Ivl -o build/vlad.o

# Leave out templated/SSE functions.
# clang -c vl/ikmeans_elkan.c -Ivl -o build/ikmeans_elkan.o
# clang -c vl/ikmeans_lloyd.c -Ivl -o build/ikmeans_lloyd.o
# clang -c vl/ikmeans_init.c -Ivl -o build/ikmeans_init.o
# clang -c vl/mathop_avx.c -Ivl -o build/mathop_avx.o
# clang -c vl/mathop_sse2.c -Ivl -o build/mathop_sse2.o
# clang -c vl/float.c -Ivl -o build/float.o
# clang -c vl/imopv_sse2.c -Ivl -o build/imopv_sse2.o

