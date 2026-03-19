function n = airindex633(P,T,RH)
% air index formula from Handbook of optical systems 
% where p is pressure in kPa, T is temperature in Celsius, and RH is relative humidity
% in percent. The equation is expected to be accurate within an estimated expanded
% uncertainty of 1.5 · 10–7.
n = 1+7.86E-6*(P/(273+T))-1.5E-11*RH*(T^2+160)
end

