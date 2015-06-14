%mining the textural features(assumes a gray image)
%x is the image
%D distance in pixels
function [statsI] = mineTexturalFeatures(x,D)

    %represents the four angles = 0, 45, 90, 135
    offsets = [0 D; -D D; -D 0; -D -D]; 
    %finding the gray-level co-occurence matrices
    glcms = graycomatrix(x,'Offset',offsets);
    %derive statistics (**entropy not considered yet)
    stats = graycoprops(glcms,{'Energy','Correlation','Homogeneity'});
    statsI = stats;
    
%     figure, plot([stats.Energy]);
%     title('Texture Energy as a function of offset');
%     ylabel('Energy')
%     pause;
%     
%     figure, plot([stats.Correlation]);
%     title('Texture Correlation as a function of offset');
%     ylabel('Correlation')
%     pause;
%     
%     figure, plot([stats.Homogeneity]);
%     title('Inverse Difference Moment as a function of offset');
%     ylabel('Inverse Difference Moment')
%     pause;

end