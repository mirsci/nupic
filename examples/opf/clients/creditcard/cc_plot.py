import numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import datetime
import matplotlib.gridspec as gridspec

# filename='credit_card_small_out.csv'
# filename='credit_card_few_features_med_out.csv'
# filename='credit_card_all_features_med_out.csv'
filename='sim_fraud_small_out.csv'

# Read data from the CSV file
convertfunc = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
data = numpy.genfromtxt(filename, delimiter=',', names=True, dtype=None,\
                        converters={0: convertfunc})
timestamp = data['timestamp']
amount = data['amount']
prediction = data['prediction']
anomalyScore = data['anomaly_score']
anomalyLikelihood = data['anomaly_likelihood']

print data['timestamp']
print data['amount']
print data['prediction']

# Plot the 2 graphs on the same figure
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3,  1])

# Define mainGraph settings
mainGraph = fig.add_subplot(gs[0, 0])
plt.title("Transactions data")
plt.ylabel('Class')
plt.xlabel('Date')
mainGraph.grid(True)

# Display the formatted timestamp on x-axis for mainGraph
ax=plt.gca()
# ax.set_xticks(timestamp)
xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
ax.xaxis_date()

# Define anomaly graph settings
# anomalyGraph = fig.add_subplot(gs[1])
# plt.title("Anomaly scores")
# plt.ylabel('Percentage')
# plt.xlabel('Date')

# Maximizes window for a QT4Agg backend, which is the default for this environment
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.tight_layout()

# dates = matplotlib.dates.date2num(timestamp)
# print dates

# Provide data to the 2 graphs
mainGraph.plot_date(timestamp, amount, label="actual", linestyle='solid', marker='.')
mainGraph.plot_date(timestamp, prediction, label="predicted", linestyle='solid', marker='.')
# mainGraph.plot_date(timestamp, anomalyScore, label="score", linestyle='solid', marker='.')
mainGraph.plot_date(timestamp, anomalyLikelihood, label="likelihood", linestyle='solid', marker='.')

# Adding legend using the above labels
mainGraph.legend(loc='upper left')

# Anomaly graph
# anomalyGraph.plot_date(timestamp, anomalyScore, label="score", linestyle='solid', marker='.')
# anomalyGraph.plot_date(timestamp, anomalyLikelihood, label="likelihood", linestyle='solid', marker='.')
# # Adding legend using the above labels
# anomalyGraph.legend(loc='upper left')


plt.savefig('foo2.png')
plt.show()