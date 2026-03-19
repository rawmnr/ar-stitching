%Lecture des positions des SSPP
    function [pos_sspp] = getPositionSspp(filename)
        fid = fopen(filename,'r');
        tline = fgetl(fid);
        i=0;
        while ischar(tline)
            i = i+1;
            temp = strsplit(tline,{'=',';'});
            pos_sspp{i,1} = str2num(temp{2});
            pos_sspp{i,2} = str2num(temp{3});
            tline = fgetl(fid);
        end
        fclose(fid);
    end

