library(ggplot2)
library(dplyr)
library(readxl)
library(readr)
library(SmartEDA)
library(GGally)
library(tidycmprsk)
library(tidyverse)
library(coin)
library(summarytools)


options(scipen=999)



amr_all <- read.csv("antibiotics_orig.csv", stringsAsFactors = T)

amr_without_genes <- read.csv("amr_without_genes.csv", stringsAsFactors = T)

amr_with_genes <- read.csv("amr_with_genes.csv", stringsAsFactors = T)

antifungals_all <- read.csv("antifungals_orig.csv", stringsAsFactors = T)

antifungals <- read.csv("antifungals.csv", stringsAsFactors = T)

?ExpCTable()
table_bact_genes <-ExpCTable(amr_with_genes, clim = 50, nlim = 50)

table_bact_nogenes <-ExpCTable(amr_without_genes, clim = 50, nlim = 50)

#Get the count of the species with genes
bact_count_genes <- amr_with_genes %>%
  group_by(Family, Species) %>%
  summarize(count = n())


write.csv(bact_count_genes, 'bact_count_genes.csv')


#Get the count of the species without genes
bact_count_nogenes <- amr_all%>%
  group_by(Family, Species) %>%
  summarize(count = n())


write.csv(bact_count_nogenes, 'bact_count_nogenes.csv')


#Get the count of the fungi species
fungi_count <- antifungals_all%>%
  group_by(Family, Species) %>%
  summarize(count = n())


write.csv(fungi_count, 'fungi_count.csv')


#Get the count of antibiotics

# Create a list of antibiotics
antibiotics_orig <- c(
  "Amikacin", "Amoxycillin.clavulanate", "Ampicillin", "Azithromycin", "Cefepime", "Cefoxitin", "Ceftazidime",
  "Ceftriaxone", "Clarithromycin", "Clindamycin", "Erythromycin", "Imipenem", "Levofloxacin", "Linezolid",
  "Meropenem", "Metronidazole", "Minocycline", "Penicillin", "Piperacillin.tazobactam", "Tigecycline",
  "Vancomycin", "Ampicillin.sulbactam", "Aztreonam", "Cefixime", "Ceftaroline", "Ceftazidime.avibactam",
  "Ciprofloxacin", "Colistin", "Daptomycin", "Doripenem", "Ertapenem", "Gentamicin", "Moxifloxacin", "Oxacillin",
  "Quinupristin.dalfopristin", "Teicoplanin", "Tetracycline", "Trimethoprim.sulfa", "Ceftolozane.tazobactam",
  "Meropenem.vaborbactam", "Aztreonam.avibactam" , "Ceftaroline.avibactam" , "Gatifloxacin", "Sulbactam", "Cefoperazone.sulbactam"
)

#For all values
amr_all_longer <- amr_all %>%
  pivot_longer(names_to = 'Antibiotics', values_to = 'MIC_Interpretation', cols = antibiotics_orig) 



antibiotics_count_nogenes <- amr_all_longer%>%
  group_by(Antibiotics, MIC_Interpretation) %>%
  summarize(count = n())

write.csv(antibiotics_count_nogenes, 'antibiotics_count.csv')

ExpCTable(species_nogenes, clim = 50, nlim = 50)



# Create a list of antibiotics
antibiotics <- c(
  "Amikacin", "Amoxycillin.clavulanate", "Ampicillin", "Azithromycin", "Cefepime", "Cefoxitin", "Ceftazidime",
  "Ceftriaxone", "Clarithromycin", "Clindamycin", "Erythromycin", "Imipenem", "Levofloxacin", "Linezolid",
  "Meropenem", "Metronidazole", "Minocycline", "Penicillin", "Piperacillin.tazobactam", "Tigecycline",
  "Vancomycin", "Ampicillin.sulbactam", "Aztreonam", "Cefixime", "Ceftaroline", "Ceftazidime.avibactam",
  "Ciprofloxacin", "Colistin", "Daptomycin", "Doripenem", "Ertapenem", "Gentamicin", "Moxifloxacin", "Oxacillin",
  "Quinupristin.dalfopristin", "Teicoplanin", "Tetracycline", "Trimethoprim.sulfa", "Ceftolozane.tazobactam",
  "Meropenem.vaborbactam", "Aztreonam.avibactam" , "Ceftaroline.avibactam" , "Gatifloxacin", "Sulbactam", "Cefoperazone.sulbactam"
)


#Convert to a long format
amr_without_genes_longer <- amr_without_genes %>%
  pivot_longer(names_to = 'Antibiotics', values_to = 'MIC_Interpretation', cols = antibiotics) %>%
  filter(MIC_Interpretation == c('Resistant', 'Intermediate', 'Susceptible'))



#Write to CSV format

write.csv(amr_without_genes_longer, 'amr_without_genes_ml.csv')


library(plyr)

# Get the unique values of each column
unique_values <- lapply(amr_without_genes_longer, unique)

# Save each unique column as a separate CSV file
for (i in seq_along(unique_values)) {
  column_name <- names(unique_values)[i]
  file_name <- paste0(column_name, ".csv")
  data <- as.data.frame(unique_values[[i]])
  write.csv(data, file = file_name, row.names = FALSE)
}




# Save the dataframe to a CSV file
write.csv(values_unique, file = "values_without_genes.csv", row.names = FALSE)




#Bacteria with genes



# Create a list of antibiotics
antibiotics1 <- c(
  "Amikacin", "Amoxycillin.clavulanate", "Ampicillin", "Cefepime", "Ceftazidime",
  "Ceftriaxone", "Imipenem", "Levofloxacin", "Linezolid",
  "Meropenem",  "Minocycline", "Piperacillin.tazobactam", "Tigecycline",
   "Ampicillin.sulbactam", "Aztreonam",  "Ceftaroline", "Ceftazidime.avibactam",
  "Ciprofloxacin", "Colistin",  "Doripenem", "Ertapenem", "Gentamicin",  
   "Trimethoprim.sulfa", "Ceftolozane.tazobactam",
  "Meropenem.vaborbactam"
)


#Convert to a long format
amr_with_genes_longer <- amr_with_genes %>%
  pivot_longer(names_to = 'Antibiotics', values_to = 'MIC_Interpretation', cols = antibiotics1) %>%
  filter(MIC_Interpretation == c('Resistant', 'Intermediate', 'Susceptible'))



#Write to CSV format

write.csv(amr_with_genes_longer, 'amr_with_genes_ml.csv')


library(plyr)

# Get the unique values of each column
unique_values <- lapply(amr_with_genes_longer, unique)

# Save each unique column as a separate CSV file
for (i in seq_along(unique_values)) {
  column_name <- names(unique_values)[i]
  file_name <- paste0(column_name, ".csv")
  data <- as.data.frame(unique_values[[i]])
  write.csv(data, file = file_name, row.names = FALSE)
}




# Save the dataframe to a CSV file
write.csv(values_unique, file = "values_without_genes.csv", row.names = FALSE)










# Create a list of antifungals
antifungal_list <- c('Anidulafungin', 'Caspofungin', 'Fluconazole', 'Micafungin', 'Voriconazole')


#Convert to a long format
amr_fungi_longer <- antifungals %>%
  pivot_longer(names_to = 'Antifungals', values_to = 'MIC_Interpretation', cols = antifungal_list) %>%
  filter(MIC_Interpretation == c('Resistant', 'Intermediate', 'Susceptible'))



#Write to CSV format

write.csv(amr_fungi_longer, 'amr_fungi_ml.csv')


library(plyr)

# Get the unique values of each column
unique_values <- lapply(amr_fungi_longer, unique)

# Save each unique column as a separate CSV file
for (i in seq_along(unique_values)) {
  column_name <- names(unique_values)[i]
  file_name <- paste0(column_name, ".csv")
  data <- as.data.frame(unique_values[[i]])
  write.csv(data, file = file_name, row.names = FALSE)
}




# Save the dataframe to a CSV file
write.csv(values_unique, file = "values_antifungals.csv", row.names = FALSE)






#Testing for collinearity

# Create contingency table of the two categorical variables
contingency_table <- table(amr_all$Family, amr_all$Species)

# Perform chi-squared test of independence
chi_square <- chisq.test(contingency_table)

# Extract the p-value from the chi-squared test
p_value <- chi_square$p.value

# Display the results
if (p_value < 0.05) {
  cat("There is evidence of a significant association between family and species (collinearity).\n")
} else {
  cat("There is no significant association between family and species (no collinearity).\n")
}

# Display the chi-squared test results
print(chi_square)



# List of gene columns
genes <- c("AMPC", "SHV", "TEM", "CTXM1", "CTXM2", "CTXM825", "CTXM9", "VEB", "PER", "GES", "ACC",
           "CMY1MOX", "CMY11", "DHA", "FOX", "ACTMIR", "KPC", "OXA", "NDM", "IMP", "VIM", "SPM", "GIM")

genes1 <- carbapenems %>%
  select(genes) %>%
  pivot_longer(names_to = 'gene_types', values_to = 'gene_subtypes', genes) %>%
  group_by(gene_types, gene_subtypes) %>%
  summarize(count = n ())

genes2 <- genes1[!is.na(genes1$gene_subtypes), ]

genes2 <- genes2[genes2$gene_subtypes != 'NEG', ]

genes2 <- genes2[genes2$gene_subtypes != '-', ]

write.csv(genes2, 'gene_subtypes.csv')



#Pairs

antibiotics_carb <- c("Amikacin", "Amoxycillin.clavulanate", "Ampicillin", "Azithromycin", "Cefepime", "Cefoxitin", "Ceftazidime", "Ceftriaxone", "Clarithromycin", "Clindamycin",
                "Erythromycin", "Imipenem", "Levofloxacin", "Linezolid", "Meropenem", "Metronidazole", "Minocycline", "Penicillin", "Piperacillin.tazobactam",
                "Tigecycline", "Vancomycin", "Ampicillin.sulbactam", "Aztreonam", "Aztreonam.avibactam", "Cefixime", "Ceftaroline", "Ceftaroline.avibactam",
                "Ceftazidime.avibactam", "Ciprofloxacin", "Colistin", "Daptomycin", "Doripenem", "Ertapenem", "Gatifloxacin", "Gentamicin", "Moxifloxacin",
                "Oxacillin", "Quinupristin.dalfopristin", "Sulbactam", "Teicoplanin", "Tetracycline", "Trimethoprim.sulfa", "Ceftolozane.tazobactam",
                "Cefoperazone.sulbactam", "Meropenem.vaborbactam")


carbapenem_genes <- amr_all %>%
  select(Year,antibiotics_carb, genes)

carbapenem_long <- carbapenem_genes %>%
  pivot_longer(names_to = 'carbapems_grp', values_to = 'Resistance_type', antibiotics_carb)

#Filter for resistant pairs
carbapenems_long_nona <- carbapenem_long %>%
  filter(Resistance_type %in% c('Resistant','Intermediate', 'Susceptible')) 

carbapenems_long_nona <- carbapenems_long_nona %>%
  pivot_longer(names_to = 'Genes_grp', values_to = 'Genes_type', genes)





#Replace NA Genes_type 
# Replace empty cells with NA
carbapenems_long_nona[carbapenems_long_nona == ""] <- NA

# Remove NAs from the Genes_type column
carbapenems_long_nona <- carbapenems_long_nona[!is.na(carbapenems_long_nona$Genes_type), ]

# Filter out rows with negative values in 'Genes_type' column
carbapenems_long_nona <- carbapenems_long_nona[carbapenems_long_nona$Genes_type != "NEG", ]

# Filter out rows with - values in 'Genes_type' column
carbapenems_long_nona <- carbapenems_long_nona[carbapenems_long_nona$Genes_type != '-', ]


summary_table_I <- carbapenems_long_nona %>%
  group_by(Resistance_type, Year) %>%
  unite(Genes_carbapenems, carbapems_grp, Genes_type, sep = " & ") 

names(summary_table_I)

summary_table_2 <- summary_table_I %>%
  group_by(Year, Genes_carbapenems, Resistance_type) %>%
  summarize(count = n())


summary(summary_table_2)

#Make it wide again based on resistance types


summary_table_3 <- summary_table_2 %>%
  pivot_wider(names_from = 'Resistance_type', values_from = 'count')

# Replace NA values with 0 in summary_table_3
summary_table_3 <- summary_table_3 %>% mutate_all(~replace(., is.na(.), 0))

summary(summary_table_3)

summary_table_3 <- summary_table_3 %>%
  mutate(Total = Resistant + Susceptible + Intermediate,
         Resistance_Percentage = (Resistant / Total) * 100)

write.csv(summary_table_3, 'carbapenem_gene_pairs_subgrp.csv')


summary_table_4 <- summary_table_3 %>%
  select(Year, Genes_carbapenems, Resistance_Percentage) %>%
  pivot_wider(names_from = Year, values_from = Resistance_Percentage)

write.csv(summary_table_4, 'carbapenem_gene_pairs_years.csv')


#Cramer's V
cramer_v_phenotype <- ExpCatStat(genotypes,Target="Phenotype",result = "Stat",clim=10,nlim=5,bins=10,Pclass=1 ,plot=FALSE,top=20,Round=2)

