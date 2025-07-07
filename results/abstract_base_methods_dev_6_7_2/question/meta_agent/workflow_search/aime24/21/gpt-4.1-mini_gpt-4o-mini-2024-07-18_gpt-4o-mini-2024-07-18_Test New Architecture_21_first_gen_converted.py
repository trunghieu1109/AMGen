async def forward_21(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction2 = "Subtask 2: Generate all line segments between any two vertices of the regular 12-gon, representing all sides and diagonals."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, generating all line segments, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction1 = "Subtask 1: Filter and label each line segment as a side or diagonal lying on the boundary of the polygon, based on output from Subtask 2."
    N = self.max_sc
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_desc=cot_sc_desc1,
        n_repeat=N
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    for idx, key in enumerate(results1['list_thinking']):
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, filtering sides/diagonals, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    logs.append(results1['subtask_desc'])

    cot_instruction0 = "Subtask 0: Determine the orientation (angle modulo 180°) for each side or diagonal to define orientation classes, based on output from Subtask 1."
    cot_agent_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results0 = await self.cot(
        subtask_id="subtask_0",
        cot_agent_desc=cot_agent_desc0
    )
    agents.append(f"CoT agent {results0['cot_agent'].id}, determining orientations, thinking: {results0['thinking'].content}; answer: {results0['answer'].content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])

    orientation_pairs = []
    # Assume results0['answer'].content contains orientation classes
    # For loop over unordered pairs of perpendicular orientation classes
    for pair_idx, orientation_pair in enumerate(orientation_pairs):
        aggregate_instruction4 = f"Subtask 4: For perpendicular orientation pair {pair_idx+1}, aggregate candidate lines into two sets for rectangle construction."
        aggregate_desc4 = {
            'instruction': aggregate_instruction4,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results4 = await self.aggregate(
            subtask_id="subtask_4",
            aggregate_desc=aggregate_desc4
        )
        agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, aggregating lines for orientation pair {pair_idx+1}, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])

        cot_instruction3 = f"Subtask 3: Generate all potential rectangles by selecting two parallel lines from each of the two orientation sets for orientation pair {pair_idx+1}."
        cot_agent_desc3 = {
            'instruction': cot_instruction3,
            'input': [taskInfo, results4['thinking'], results4['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
        }
        results3 = await self.cot(
            subtask_id="subtask_3",
            cot_agent_desc=cot_agent_desc3
        )
        agents.append(f"CoT agent {results3['cot_agent'].id}, generating rectangles for orientation pair {pair_idx+1}, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])

        cot_agent_instruction5 = f"Subtask 5: Apply validation criteria to filter out degenerate or overlapping rectangle candidates for orientation pair {pair_idx+1}."
        cot_agent_desc5 = {
            'instruction': cot_agent_instruction5,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        results5 = await self.answer_generate(
            subtask_id="subtask_5",
            cot_agent_desc=cot_agent_desc5
        )
        agents.append(f"AnswerGenerate agent {results5['answer_generate_agent'].id}, validating rectangles for orientation pair {pair_idx+1}, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
        sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
        logs.append(results5['subtask_desc'])

    condition_complexity = True
    if condition_complexity:
        # True branch: detailed geometric intersection checks
        # Start true branch
        # (No subtasks explicitly defined, so just placeholder logic)
        # End true branch
        pass
    else:
        # False branch: skip extended checks
        # Start false branch
        # (No subtasks explicitly defined, so just placeholder logic)
        # End false branch
        pass

    formatter_desc6 = {
        'instruction': "Subtask 6: Identify and clarify each valid rectangle as four distinct vertices and ensure no duplicates.",
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"],
        'format': 'short and concise, without explaination'
    }
    results6 = await self.specific_format(
        subtask_id="subtask_6",
        formatter_desc=formatter_desc6
    )
    agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, identifying and clarifying rectangles, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    review_instruction7 = "Subtask 7: Review the list of validated rectangles to confirm completeness, uniqueness, and correctness."
    review_desc7 = {
        'instruction': review_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results7 = await self.review(
        subtask_id="subtask_7",
        review_desc=review_desc7
    )
    agents.append(f"Review agent {results7['review_agent'].id}, reviewing rectangles, feedback: {results7['thinking'].content}; correct: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: feedback - {results7['thinking'].content}; correct - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    formatter_desc8 = {
        'instruction': "Subtask 8: Enhance clarity and coherence of the rectangle list, preparing for final presentation.",
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"],
        'format': 'short and concise, without explaination'
    }
    results8 = await self.specific_format(
        subtask_id="subtask_8",
        formatter_desc=formatter_desc8
    )
    agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, enhancing clarity, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    formatter_desc9 = {
        'instruction': "Subtask 9: Format the final count as a single integer, per output requirements.",
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 8", "answer of subtask 8"],
        'format': 'short and concise, without explaination'
    }
    results9 = await self.specific_format(
        subtask_id="subtask_9",
        formatter_desc=formatter_desc9
    )
    agents.append(f"SpecificFormat agent {results9['formatter_agent'].id}, formatting final count, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
    return final_answer, logs
