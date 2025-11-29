def generate_queue_insights(self, queue_data):
    insights = {
        'queue_length': len(queue_data['people']),
        'estimated_wait_time': self.estimate_wait_time(queue_data),
        'queue_growth_trend': self.analyze_growth_trend(queue_data),
        'recommended_staff_allocation': self.suggest_staffing(queue_data)
    }
    return insights