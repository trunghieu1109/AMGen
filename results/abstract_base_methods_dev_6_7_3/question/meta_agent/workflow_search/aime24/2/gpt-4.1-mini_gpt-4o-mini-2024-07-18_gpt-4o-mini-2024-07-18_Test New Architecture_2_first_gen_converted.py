async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction0 = "Subtask 1: Analyze the problem of coloring vertices of a regular octagon with red or blue independently and the condition involving rotations mapping blue vertices to original red vertices. Define variables and formal relationships to represent the problem."
    cot_agent_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc0
    )
    agents.append(f"CoT agent {results0['cot_agent'].id}, analyzing problem setup, thinking: {results0['thinking'].content}; answer: {results0['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])

    cot_sc_instruction1 = "Subtask 2: Using the formal relationships from Subtask 1, consider all possible rotations and colorings, and calculate the probability that a rotation exists mapping blue vertices to original red vertices. Use self-consistency to consider multiple cases."
    N = self.max_sc
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results1 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc1,
        n_repeat=N
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    for idx in range(len(results1['list_thinking'])):
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, considering rotations and colorings, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction2 = "Subtask 3: Based on outputs from Subtask 1 and 2, filter valid scenarios where the rotation condition is met and the probability can be computed correctly."
    critic_instruction2 = "Please review the filtering of valid scenarios and provide limitations or corrections."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results0['thinking'], results0['answer'], results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, filtering valid rotation scenarios, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][i].content}; correction: {results2['list_correct'][i].content}")
        if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
            agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining answer, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    candidates = []
    for rotation_idx in range(8):
        generate_instruction = f"Subtask 4: Generate initial candidate solution for rotation {rotation_idx} by reasoning about color mappings and probability contributions."
        generate_desc = {
            'instruction': generate_instruction,
            'input': [taskInfo, results2['thinking'], results2['answer']],
            'temperature': 0.0,
            'context': ["user query", "filtered valid scenarios"]
        }
        results_candidate = await self.answer_generate(
            subtask_id=f"subtask_4_{rotation_idx + 1}",
            cot_agent_desc=generate_desc
        )
        agents.append(f"AnswerGenerate agent {results_candidate['cot_agent'].id}, generating candidate for rotation {rotation_idx}, thinking: {results_candidate['thinking'].content}; answer: {results_candidate['answer'].content}")
        sub_tasks.append(f"Sub-task 4_{rotation_idx + 1} output: thinking - {results_candidate['thinking'].content}; answer - {results_candidate['answer'].content}")
        logs.append(results_candidate['subtask_desc'])
        candidates.append(results_candidate)

    aggregate_instruction = "Subtask 5: Aggregate candidate solutions from all rotations, evaluate consistency, and select the most coherent probability value."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + [c['answer'] for c in candidates],
        'temperature': 0.0,
        'context': ["user query", "candidate solutions from rotations"]
    }
    results5 = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results5['aggregate_agent'].id}, aggregating candidates, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    identify_instruction = "Subtask 6: Identify and clarify units and validate the aggregated probability value for correctness and completeness."
    identify_desc = {
        'instruction': identify_instruction,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "aggregated probability"]
    }
    results6 = await self.specific_format(
        subtask_id="subtask_6",
        formatter_desc=identify_desc
    )
    agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, clarifying units and validating probability, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    review_instruction = "Subtask 7: Review the clarified and validated probability value to ensure accuracy and consistency."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "clarified probability"]
    }
    results7 = await self.review(
        subtask_id="subtask_7",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results7['review_agent'].id}, reviewing probability value, feedback: {results7['thinking'].content}; correct: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: feedback - {results7['thinking'].content}; correct - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    if results7['answer'].content.lower() in ['yes', 'correct', 'true']:
        true_branch_instruction = "Subtask 8: Finalize the probability calculation and prepare the final answer as an integer sum m+n."
        true_branch_desc = {
            'instruction': true_branch_instruction,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "reviewed probability"]
        }
        results8 = await self.answer_generate(
            subtask_id="subtask_8",
            cot_agent_desc=true_branch_desc
        )
        agents.append(f"AnswerGenerate agent {results8['cot_agent'].id}, finalizing answer, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])
        final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
        return final_answer, logs
    else:
        false_branch_instruction = "Subtask 9: Reassess and refine the probability calculation due to review feedback indicating issues."
        false_branch_desc = {
            'instruction': false_branch_instruction,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "review feedback"]
        }
        results9 = await self.answer_generate(
            subtask_id="subtask_9",
            cot_agent_desc=false_branch_desc
        )
        agents.append(f"AnswerGenerate agent {results9['cot_agent'].id}, refining answer, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])
        final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
        return final_answer, logs
