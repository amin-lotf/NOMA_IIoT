% 5 sbs, 4 sc,  0.6 arrival rate, 0.3 type probabitly
n_runs = 1;
font_size_reward=19;
x_limit = 120*24*60;
x_limit_reward = 20000;
marker_size=8;
face_alpha=0.5;
line_width=2.5;
mappo1_marker = '-o';
mappo2_marker = '--s';
mappo3_marker = ':^';
mappo4_marker = '-v';

edge_alpha= 0.18;
mappo1_color='#0072B2';
mappo2_color='#f53242';
mappo3_color='#a305ff';
mappo4_color='#2cdb04';
mappo5_color='#f2906f';
mappo6_color='#f39f6f';
load_info

legend_names = {
    '\textbf{NOMA (}\boldmath$x^{\textbf{max}}=3,$ \textbf{ 48 IDs)}',...
     '\textbf{NOMA (}\boldmath$x^{\textbf{max}}=2,$ \textbf{ 32 IDs)}', ...
    '\textbf{OMA}$\phantom{10}$ \textbf{(}\boldmath$x^{\textbf{max}}=1,$ \textbf{ 16 IDs)}'
};

x_labels={'\textbf{EP}';'\textbf{Processed tasks}'};

bar_legend_names=categorical(x_labels);
ep=[3.25;3.58;4.06];
bits=[70.22;45.36;23.63];
p=[1.81;4.06;6.33];


ep_i=(ep(1)-ep(3))/ep(1)
bits_i=(bits(3)-bits(1))/bits(1)
h15=figure(15);
clf(h15,'reset');
x = bar_legend_names;
x = reordercats(x,cellstr(x)');

nil = [0; 0 ; 0 ];
b=bar(x, [ep nil]);
for i = 1:numel(b)
    % Get the X- and Y-coordinates at the top of each bar
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;
    
    % Convert the YData into a cell array of strings (or use string(...) directly)
    labels = string(b(i).YData);
    labels(b(i).YData == 0) = "";
    % Place text centered horizontally and just above each bar
    text(xtips, ytips, labels, ...
         'HorizontalAlignment','center', ...
         'VerticalAlignment','bottom',...
         'FontSize',11);
end
ylabel("\textbf{EP (Mbits/J)}",'Interpreter','latex','FontSize', font_size_print);
 ylim([0 6.0]);
yyaxis right
b(1).FaceColor = '#0072B2';
b(2).FaceColor = '#D55E00';
b(3).FaceColor = '#009E73';
b=bar(x, [nil bits],'FaceColor','flat');
for i = 1:numel(b)
    % Get the X- and Y-coordinates at the top of each bar
    xtips = b(i).XEndPoints;
    ytips = b(i).YEndPoints;

    % Convert the YData into a cell array of strings (or use string(...) directly)
    labels = string(b(i).YData);
    labels(b(i).YData == 0) = "";
    % Place text centered horizontally and just above each bar
    text(xtips, ytips, labels, ...
         'HorizontalAlignment','center', ...
         'VerticalAlignment','bottom',...
         'FontSize',11);
end
ylabel("\textbf{Processed tasks (Mbits)}",'Interpreter','latex','FontSize', font_size_print);
xlabel("\textbf{Metric}",'Interpreter','latex','FontSize', font_size_reward);
grid on
b(1).FaceColor = '#0072B2';
b(2).FaceColor = '#D55E00';
b(3).FaceColor = '#009E73';
set(gca,'ycolor','k') 
set(gca,'FontSize',font_size_print_m)
set(gca, 'XTickLabel', x_labels, 'TickLabelInterpreter', 'latex');
ylim([0 100]);
legend(legend_names,'Interpreter','latex','Location','northeast','FontSize', font_size_print_m,'NumColumns',1);
% saveas(h15,'graphs/ec_rate_2_bar.png');