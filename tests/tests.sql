create table tests(
    requirement text,
    program text,
    input text,
    output text
);

insert into tests values('image_processing', 'process_keypoints', '1', '1');
insert into tests values('video_processing', 'process_frames', '2', '2');

select * from tests;
