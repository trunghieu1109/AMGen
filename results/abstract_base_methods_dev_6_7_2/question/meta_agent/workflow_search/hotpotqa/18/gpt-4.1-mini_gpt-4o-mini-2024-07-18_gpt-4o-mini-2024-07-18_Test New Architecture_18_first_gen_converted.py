async def forward_18(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_cultures = []
    # Control Flow 0: start sequential
    # Control Flow 1: start loop - generate multiple candidate outputs
    for i in range(self.max_sc):
        cot_sc_instruction = f"Sub-task {i+1}: Systematically generate a candidate European culture origin of the trophy awarded to the winner of the University of Idaho Vandals vs. University of Montana Grizzlies football game."
        cot_sc_desc = {
            'instruction': cot_sc_instruction,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results = await self.sc_cot(
            subtask_id=f"subtask_{i+1}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results['cot_agent'][0].id}, generating candidate culture, thinking: {results['list_thinking'][0]}; answer: {results['list_answer'][0]}")
        sub_tasks.append(f"Sub-task {i+1} output: thinking - {results['list_thinking'][0]}; answer - {results['list_answer'][0]}")
        logs.append(results['subtask_desc'])
        candidate_cultures.append(results['list_answer'][0])
    # Control Flow 2: end loop
    # Stage 1: consolidate multiple inputs
    aggregate_instruction = "Sub-task {}: Integrate multiple candidate cultural origins by evaluating their consistency and synthesizing them into a single coherent output.".format(self.max_sc + 1)
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_cultures,
        'temperature': 0.0,
        'context': ["user query", "candidate cultural origins"]
    }
    results_agg = await self.aggregate(
        subtask_id=f"subtask_{self.max_sc + 1}",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_agg['aggregate_agent'].id}, consolidating candidate cultures, thinking: {results_agg['thinking'].content}; answer: {results_agg['answer'].content}")
    sub_tasks.append(f"Sub-task {self.max_sc + 1} output: thinking - {results_agg['thinking'].content}; answer - {results_agg['answer'].content}")
    logs.append(results_agg['subtask_desc'])
    # Stage 1: validate consolidated output
    review_instruction = "Sub-task {}: Review the consolidated cultural origin output to confirm its accuracy, completeness, validity, and correctness.".format(self.max_sc + 2)
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_agg['thinking'], results_agg['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of consolidated output", "answer of consolidated output"]
    }
    results_review = await self.review(
        subtask_id=f"subtask_{self.max_sc + 2}",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing consolidated culture, feedback: {results_review['thinking'].content}; correct: {results_review['answer'].content}")
    sub_tasks.append(f"Sub-task {self.max_sc + 2} output: feedback - {results_review['thinking'].content}; correct - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    programmer_instruction = "Sub-task {}: Verify and validate the reviewed cultural origin output by generating code or structured logic to confirm its correctness.".format(self.max_sc + 3)
    programmer_desc = {
        'instruction': programmer_instruction,
        'input': [taskInfo, results_agg['answer'], results_review['answer']],
        'temperature': 0.0,
        'context': ["user query", "consolidated answer", "review feedback"],
        'entry_point': "validate_cultural_origin"
    }
    results_prog = await self.programmer(
        subtask_id=f"subtask_{self.max_sc + 3}",
        programmer_desc=programmer_desc
    )
    agents.append(f"Programmer Agent {results_prog['programmer_agent'].id}, validating consolidated culture, thinking: {results_prog['thinking'].content}; answer: {results_prog['answer'].content}, executing results: {results_prog['exec_result']}")
    sub_tasks.append(f"Sub-task {self.max_sc + 3} output: thinking - {results_prog['thinking'].content}; answer - {results_prog['answer'].content}; output - {results_prog['exec_result']}")
    logs.append(results_prog['subtask_desc'])
    cot_instruction_final = "Sub-task {}: Based on the reviewed and validated outputs, provide a final coherent answer identifying the European culture from which the trophy is derived.".format(self.max_sc + 4)
    cot_agent_desc_final = {
        'instruction': cot_instruction_final,
        'input': [taskInfo, results_agg['answer'], results_review['answer'], results_prog['answer']],
        'temperature': 0.0,
        'context': ["user query", "aggregated answer", "review feedback", "programmer validation"]
    }
    results_final = await self.cot(
        subtask_id=f"subtask_{self.max_sc + 4}",
        cot_agent_desc=cot_agent_desc_final
    )
    agents.append(f"CoT agent {results_final['cot_agent'].id}, final reasoning, thinking: {results_final['thinking'].content}; answer: {results_final['answer'].content}")
    sub_tasks.append(f"Sub-task {self.max_sc + 4} output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    logs.append(results_final['subtask_desc'])
    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs