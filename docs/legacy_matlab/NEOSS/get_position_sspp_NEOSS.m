function [pos_sspp] = get_position_sspp_NEOSS(filename)

    fid = fopen(filename,'r');
    tline = fgetl(fid);
    i = 0;
    while ischar(tline)
        i = i+1;
        temp = strsplit(tline,{';','='});
        for j = 1:length(temp)  
            pos_sspp{i,j} = str2num(temp{j});  
        end
        pos_sspp{i,1}=temp{1};
        tline = fgetl(fid);
    end
    fclose(fid);
    
end