function [] =Excel_ErrorBudget_filling(Mirror_map,ErrorSource,Excelname_towrite)
%Excel_ErrorBudget_filling writes the results in the Error Budget Excel
%file

%Loading project definitions
% ErrorBudget_project_def

%Reading the Excel file
Error_budget_Excel = readtable(Excelname_towrite);


%Chosing the polynomials type analysis 
if length(Mirror_map.Zcoeff)>-1
    COEFF = 1;
end

switch COEFF
    
    case 1      %Zernikes
        
            %From coeff to nm RMS - Zernikes
            FromnmRMS2Zi = [ sqrt(3) sqrt(6) sqrt(6) sqrt(8) sqrt(8) sqrt(5) sqrt(8) sqrt(8) sqrt(10) sqrt(10) sqrt(12) sqrt(12) sqrt(7)...
                            sqrt(10) sqrt(10) sqrt(12) sqrt(12) sqrt(14) sqrt(14) sqrt(16) sqrt(16) sqrt(9) sqrt(12) sqrt(12) sqrt(14) sqrt(14)...
                            sqrt(16) sqrt(16) sqrt(18) sqrt(18) sqrt(20) sqrt(20) sqrt(11) sqrt(13) 1];
             
            FromZi2nmRMS = FromnmRMS2Zi.^(-1);
            
            %Sensibilities vector building
            S_vec = Mirror_map.Zcoeff';
            S_vec = S_vec(4:length(S_vec));     %Removing Piston, TiltX and TiltY
            S_vec = [S_vec Mirror_map.MFHF];    %Adding MF/HF
            S_vec = S_vec.*FromZi2nmRMS;
 
            for it=1:length(S_vec)
                
                eval([ 'Error_budget_Excel.' ErrorSource.label '{6+it}=S_vec(0+it);' ]);
                
            end
            
          
            
            %Value used for the error estimation
            eval([ 'Error_budget_Excel.' ErrorSource.label '(1)={' num2str(ErrorSource.variation,'%10.8e\n') '};' ]);
            
            %Coeff and contingency margin
            eval([ 'Error_budget_Excel.' ErrorSource.label '(43)={' num2str(ErrorSource.coeff,'%10.8e\n') '};' ]);
            eval([ 'Error_budget_Excel.' ErrorSource.label '(44)={' num2str(ErrorSource.ContingencyMargin,'%10.8e\n') '};' ]);
                        
            %The needed formulas
            
            %Coeff X
            eval([ 'CoeffX = Error_budget_Excel.' ErrorSource.label '{43}.*Error_budget_Excel.' ErrorSource.label '{44};']);
            eval([ 'Error_budget_Excel.' ErrorSource.label '(4)={' num2str(CoeffX,'%10.8e\n') '};' ]);
            
            %PtV 1/2
            eval([ 'PtV12 = Error_budget_Excel.' ErrorSource.label '{43}.*Error_budget_Excel.' ErrorSource.label '{1};']);
            eval([ 'Error_budget_Excel.' ErrorSource.label '(5)={' num2str(PtV12,'%10.8e\n') '};']);
            
            %Unit
            eval([ 'Error_budget_Excel.' ErrorSource.label '(6)={' native2unicode(39) ErrorSource.unit native2unicode(39) '};']);
            
            %Average or standard deviation contributor
            
            if ErrorSource.type == 1     %average
                eval([ 'Error_budget_Excel.' ErrorSource.label '(2)=' 'Error_budget_Excel.' ErrorSource.label '(4);' ]);
                eval([ 'Error_budget_Excel.' ErrorSource.label '(3)={0};' ]);
            else                         %standard deviation
                eval([ 'Error_budget_Excel.' ErrorSource.label '(2)={0};' ]);
                eval(['junk = Error_budget_Excel.' ErrorSource.label '{4}/ sqrt(3);']);
                eval([ 'Error_budget_Excel.' ErrorSource.label '(3)={' num2str(junk,'%10.8e\n') '};' ]);
            end
                
            
            %Writing the new table
            writetable(Error_budget_Excel,Excelname_towrite,'Sheet',1,'Range','B1','WriteVariableNames',true,'WriteRowNames',true);

end

end

