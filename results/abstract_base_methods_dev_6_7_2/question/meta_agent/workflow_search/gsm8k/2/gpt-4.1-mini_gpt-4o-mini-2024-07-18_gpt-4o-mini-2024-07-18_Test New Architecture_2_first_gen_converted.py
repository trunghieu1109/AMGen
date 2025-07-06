async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    lengths = [150, 490]
    costs = []
    # Control Flow 0: start sequential
    # Control Flow 1: start loop for repaving cost calculations
    for i, length in enumerate(lengths, start=1):
        cot_instruction = f"Sub-task {i}: Calculate the repaving cost for a street {length} meters long at $194 per meter, with context from taskInfo"
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results = await self.sc_cot(
            subtask_id=f"subtask_{i}",
            cot_sc_desc=cot_agent_desc,
            n_repeat=self.max_sc
        )
        agents.append(f"CoT-SC agent {results['cot_agent'][0].id}, calculating cost for street length {length}, thinking: {results['list_thinking'][0]}; answer: {results['list_answer'][0]}")
        sub_tasks.append(f"Sub-task {i} output: thinking - {results['thinking'].content}; answer - {results['answer'].content}")
        logs.append(results['subtask_desc'])
        costs.append(results['answer'].content)
    # Control Flow 2: end loop
    # Stage 1: consolidate the two repaving costs by calculating the difference
    aggregate_instruction = "Sub-task 3: Calculate how much more it costs to repave Lewis' street compared to Monica's street, based on the two calculated costs."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + costs,
        'temperature': 0.0,
        'context': ["user query", "costs from subtask 1 and 2"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, calculating cost difference, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    # Stage 1: validate the calculated cost difference
    review_instruction = "Sub-task 4: Review the calculated cost difference to ensure accuracy and correctness."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results4['review_agent'].id}, reviewing cost difference, feedback: {results4['thinking'].content}; correct: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: feedback - {results4['thinking'].content}; correct - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs