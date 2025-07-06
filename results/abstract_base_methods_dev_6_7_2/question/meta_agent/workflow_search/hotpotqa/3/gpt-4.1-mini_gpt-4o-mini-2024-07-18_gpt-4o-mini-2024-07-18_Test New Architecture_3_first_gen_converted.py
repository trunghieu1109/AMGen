async def forward_3(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start sequential
    
    # Control Flow 1: start loop for candidate generation
    candidate_results = []
    for i in range(self.max_sc):
        cot_instruction = f"Sub-task {i*2+1}: Generate candidate animation comedy created by the creator of 'Wallace and Gromit' that matches the description of animated zoo animals with a soundtrack of people talking about their homes. Iteration {i+1}."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results_cot = await self.cot(
            subtask_id=f"subtask_{i*2+1}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results_cot['cot_agent'].id}, generating candidate animation comedy, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
        sub_tasks.append(f"Sub-task {i*2+1} output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
        logs.append(results_cot['subtask_desc'])
        
        cot_sc_instruction = f"Sub-task {i*2+2}: Refine and consider multiple possible candidate animation comedies based on previous output to ensure consistency and coverage. Iteration {i+1}."
        cot_sc_desc = {
            'instruction': cot_sc_instruction,
            'input': [taskInfo, results_cot['thinking'], results_cot['answer']],
            'temperature': 0.7,
            'context': ["user query", "thinking of previous subtask", "answer of previous subtask"]
        }
        results_sc = await self.sc_cot(
            subtask_id=f"subtask_{i*2+2}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=self.max_sc
        )
        for idx in range(self.max_sc):
            agents.append(f"CoT-SC agent {results_sc['cot_agent'][idx].id}, refining candidate animation comedy, thinking: {results_sc['list_thinking'][idx]}; answer: {results_sc['list_answer'][idx]}")
        sub_tasks.append(f"Sub-task {i*2+2} output: thinking - {results_sc['thinking'].content}; answer - {results_sc['answer'].content}")
        logs.append(results_sc['subtask_desc'])
        candidate_results.append(results_sc['answer'].content)
    
    # Control Flow 2: end loop
    
    # Stage 1: consolidate multiple inputs
    aggregate_instruction = "Sub-task 4: From the generated candidate animation comedies, aggregate these solutions and return the consistent and best solution that matches the description of animated zoo animals with a soundtrack of people talking about their homes."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_results,
        'temperature': 0.0,
        'context': ["user query", "candidate animation comedies"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, consolidating candidate animation comedies, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    
    # Stage 1: validate consolidated output
    review_instruction = "Sub-task 5: Review the consolidated animation comedy output to validate its accuracy, completeness, and correctness against the criteria of matching animated zoo animals with a soundtrack of people talking about their homes."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of consolidated output", "answer of consolidated output"]
    }
    results_review = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing consolidated output, feedback: {results_review['thinking'].content}; correct: {results_review['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: feedback - {results_review['thinking'].content}; correct - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    
    final_answer = await self.make_final_answer(results_review['thinking'], results_review['answer'], sub_tasks, agents)
    return final_answer, logs