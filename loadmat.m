clear all;
close all;
C = 3e8;

NumFreq = 401;
NumAzimuth = 512;
NumElevation = 512;
B = 2e10;
dx = C/2/B;
dy = dx;
XBegin = 0;
YBegin = 0;
X = XBegin + (-dx*NumAzimuth/2:dx:dx*(NumAzimuth/2-1));
Y = YBegin + (-dy*NumFreq/2:dy:dy*(NumFreq/2-1));
folder_path = './TEST_DATA/DATA_01/';
save_ev_path = './test/ev/'
save_eh_path = './test/eh/'
dir_info = dir(folder_path);
for i = 1:length(dir_info)
    % 判断是否为目录
    if (dir_info(i).isdir && ~strcmp(dir_info(i).name, '.') && ~strcmp(dir_info(i).name, '..'))
        subfolder_path = fullfile(folder_path, dir_info(i).name);
        save_ev_subpath = fullfile(save_ev_path, dir_info(i).name);
        save_eh_subpath = fullfile(save_eh_path, dir_info(i).name);
        % 获取子目录下的所有mat文件
        mat_files = dir(fullfile(subfolder_path, '*.mat'));
        for j = 1:length(mat_files)
            % 读取mat文件中的数据
            tempname = strsplit(mat_files(j).name, '.');
            file_name = char(tempname(1));
            file_path = fullfile(subfolder_path, mat_files(j).name);

            load(file_path);
            ISAREh=fftshift(ifftn(frame_Eh)); 
            ISAREv=fftshift(ifftn(frame_Ev)); 
            % Window = hamming(NumFreq);
            % Window_T = reshape(Window, [1,NumFreq]);
            % for i = 1:1:NumAzimuth
            %     ISAREhWin(:,i) = ISAREh(:,i) .* Window;
            %     ISAREvWin(:,i) = ISAREv(:,i) .* Window;
            % end
            
%             figure();
%             subplot(2,3,1);
%             imagesc(X,Y,abs(ISAREh)); 
%             % colormap(1-gray); 
%             colorbar
%             set(gca,'FontName', 'Arial', 'FontSize',12,'FontWeight', 'Bold');
%             xlabel('Range [m] (Y)'); ylabel('Cross - range [m] (X) ');
%             title("isar-Eh");
%             hold on;
%             
% 
%             
%             subplot(2,3,3);
%             Isardbeh = 10*log10(abs(ISAREh));
%             imagesc(X,Y,Isardbeh); 
%             % colormap(1-gray); 
%             colorbar
%             set(gca,'FontName', 'Arial', 'FontSize',12,'FontWeight', 'Bold');
%             xlabel('Range [m] (Y)'); ylabel('Cross - range [m] (X)');
%             title("isar-Eh-db");
%             hold on;
%             
%             subplot(2,3,4);
%             imagesc(X,Y,abs(ISAREv)); 
%             % colormap(1-gray); 
%             colorbar
%             set(gca,'FontName', 'Arial', 'FontSize',12,'FontWeight', 'Bold');
%             xlabel('Range [m] (Y)'); ylabel('Cross - range [m] (X) ');
%             title("isar-Ev");
%             hold on;
%             
%             
%             subplot(2,3,6);
%             Isardbev = 10*log10(abs(ISAREv));
%             imagesc(X,Y,Isardbev); 
%             % colormap(1-gray); 
%             colorbar
%             set(gca,'FontName', 'Arial', 'FontSize',12,'FontWeight', 'Bold');
%             xlabel('Range [m] (Y)'); ylabel('Cross - range [m] (X)');
%             title("isar-Ev-db");
%             hold on;
            if ~exist(save_ev_subpath, 'dir')
    % 如果文件路径不存在，则创建路径
                mkdir(save_ev_subpath);
            end
            if ~exist(save_eh_subpath, 'dir')
    % 如果文件路径不存在，则创建路径
                mkdir(save_eh_subpath);
            end
            abs_ev = mat2gray(abs(ISAREv));
            abs_eh = mat2gray(abs(ISAREh));
%             abs_ev_norm = normalize(abs_ev, 'range'); 
            file_ev_path = fullfile(save_ev_subpath, [file_name,'.png']);
            file_eh_path = fullfile(save_eh_subpath, [file_name,'.png']);
            imwrite(abs_ev,file_ev_path, 'png');
            imwrite(abs_eh,file_eh_path, 'png');
%             file_ev_path = fullfile(save_ev_subpath, [file_name,'.bmp']);
%             file_eh_path = fullfile(save_eh_subpath, [file_name,'.bmp']);
%             imwrite(abs(ISAREv),file_ev_path, 'bmp');
%             imwrite(abs(ISAREh),file_eh_path, 'bmp');
%             
            % 处理mat_data中的数据
            % ...
        end
    end

end