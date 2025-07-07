async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_expansions = []
    # Control Flow 0: start sequential
    # Control Flow 1: start loop for candidate generation
    for i in range(3):
        cot_instruction = f"Subtask {i+1}: Systematically generate a candidate expansion of the new acronym adopted by VIVA Media AG after its 2004 name change by applying structured reasoning and searching relevant sources."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results = await self.cot(
            subtask_id=f"subtask_{i+1}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results['cot_agent'].id}, generating candidate expansion, thinking: {results['thinking'].content}; answer: {results['answer'].content}")
        sub_tasks.append(f"Subtask {i+1} output: thinking - {results['thinking'].content}; answer - {results['answer'].content}")
        logs.append(results['subtask_desc'])
        candidate_expansions.append(results['answer'].content)
    # Control Flow 2: end loop
    # Stage 1: consolidate multiple inputs
    aggregate_instruction = "Subtask 4: Integrate multiple candidate acronym expansions by evaluating their consistency and synthesizing them into a single coherent output."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_expansions,
        'temperature': 0.0,
        'context': ["user query", "candidate expansions from subtasks 1-3"]
    }
    results_agg = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_agg['aggregate_agent'].id}, consolidating candidate expansions, thinking: {results_agg['thinking'].content}; answer: {results_agg['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results_agg['thinking'].content}; answer - {results_agg['answer'].content}")
    logs.append(results_agg['subtask_desc'])
    # Stage 1: validate consolidated output
    review_instruction = "Subtask 5: Evaluate the consolidated acronym expansion against established criteria to confirm its accuracy, completeness, validity, and correctness."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_agg['thinking'], results_agg['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results_review = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing consolidated expansion, feedback: {results_review['thinking'].content}; correct: {results_review['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results_review['thinking'].content}; correct - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    programmer_instruction = "Subtask 6: Based on the review, generate a final validated and executable explanation or code snippet that confirms the consolidated acronym expansion's correctness."
    programmer_desc = {
        'instruction': programmer_instruction,
        'input': [taskInfo, results_agg['thinking'], results_agg['answer'], results_review['thinking'], results_review['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking and answer of subtask 4", "feedback and correct of subtask 5"]
    }
    results_prog = await self.programmer(
        subtask_id="subtask_6",
        programmer_desc=programmer_desc
    )
    agents.append(f"Programmer agent {results_prog['programmer_agent'].id}, validating consolidated expansion, thinking: {results_prog['thinking'].content}; answer: {results_prog['answer'].content}; output: {results_prog['exec_result']}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results_prog['thinking'].content}; answer - {results_prog['answer'].content}; output - {results_prog['exec_result']}")
    logs.append(results_prog['subtask_desc'])
    final_answer = await self.make_final_answer(results_prog['thinking'], results_prog['answer'], sub_tasks, agents)
    return final_answer, logs