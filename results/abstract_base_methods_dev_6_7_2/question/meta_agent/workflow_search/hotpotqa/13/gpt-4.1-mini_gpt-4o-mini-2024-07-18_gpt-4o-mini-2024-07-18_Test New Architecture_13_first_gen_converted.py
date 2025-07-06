async def forward_13(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    teen_titans_spinoffs = ["Teen Titans Go!", "Young Justice", "Justice League Action", "Teen Titans (2003 series)"]
    actress_name = "Canadian-American actress who voiced Juliet Starling"
    voice_roles_results = []
    # Control Flow 0: start sequential
    # Control Flow 1: start loop over spinoff series
    for idx, spinoff in enumerate(teen_titans_spinoffs, start=1):
        cot_instruction = f"Sub-task {idx}: For the Teen Titans spinoff series '{spinoff}', check if the actress {actress_name} has done voice roles in it, using chain-of-thought reasoning."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo, spinoff, actress_name],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results_cot = await self.cot(
            subtask_id=f"subtask_{idx}_cot",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results_cot['cot_agent'].id}, analyzing spinoff '{spinoff}', thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
        sub_tasks.append(f"Sub-task {idx} CoT output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
        logs.append(results_cot['subtask_desc'])
        cot_sc_instruction = f"Sub-task {idx}: Based on the CoT output, consider multiple possible voice role involvements of the actress in '{spinoff}' with self-consistency reasoning."
        cot_sc_desc = {
            'instruction': cot_sc_instruction,
            'input': [taskInfo, spinoff, actress_name, results_cot['thinking'], results_cot['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask {idx} CoT", "answer of subtask {idx} CoT"]
        }
        results_sc = await self.sc_cot(
            subtask_id=f"subtask_{idx}_sc_cot",
            cot_sc_desc=cot_sc_desc,
            n_repeat=self.max_sc
        )
        for i in range(self.max_sc):
            agents.append(f"CoT-SC agent {results_sc['cot_agent'][i].id}, considering voice roles in '{spinoff}', thinking: {results_sc['list_thinking'][i]}; answer: {results_sc['list_answer'][i]}")
        sub_tasks.append(f"Sub-task {idx} SC-CoT output: thinking - {results_sc['thinking'].content}; answer - {results_sc['answer'].content}")
        logs.append(results_sc['subtask_desc'])
        voice_roles_results.append(results_sc['answer'].content)
    # Control Flow 2: end loop
    # Stage 1: consolidate multiple inputs
    aggregate_instruction = "Sub-task 4: Consolidate the list of Teen Titans spinoff series where the actress has voice roles, aggregating all candidate outputs from previous subtasks."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + voice_roles_results,
        'temperature': 0.0,
        'context': ["user query", "solutions generated from subtask 1 to subtask n"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, consolidating voice roles, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    # Stage 1: validate consolidated output
    review_instruction = "Sub-task 5: Review the consolidated list of Teen Titans spinoff series for accuracy and completeness."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results_review = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing consolidated output, feedback: {results_review['thinking'].content}; correct: {results_review['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: feedback - {results_review['thinking'].content}; correct - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    # Final answer processing
    final_answer = await self.make_final_answer(results_review['thinking'], results_review['answer'], sub_tasks, agents)
    return final_answer, logs