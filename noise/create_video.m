clc; clear;
pkg load video;
pkg load image;

fr = 1023; % Numero de fotogramas

% Crear el nuevo video
video_write = VideoWriter('costarica_restored.mp4');
% Abrir el archivo del video nuevo
open(video_write);

% Procesar los fotogramas del video
disp('Processing...');
for k = 0:fr
  k
  % Leer el fotograma
  fname = sprintf('filtered_omp_gpu/frame%d.png', k); % Nombre del archivo
  frame = imread(fname);
  % Escribir cada fotograma del video
  writeVideo(video_write, frame);
end

close(video_write);
disp('Finished');