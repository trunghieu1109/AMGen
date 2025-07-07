async def forward_21(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction2 = "Subtask 2: Generate all line segments between any two vertices of the regular 12-gon, including wrap-around sides, representing all sides and diagonals."
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

    cot_instruction1 = "Subtask 1: From the generated segments, identify and label which segments connect consecutive vertices (including wrap-around) as sides, and label the rest as diagonals."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, labeling sides and diagonals, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction0 = "Subtask 0: Determine the orientation (angle modulo 180°) for each side or diagonal segment, verifying angle calculations against known properties of the regular dodecagon to ensure correctness."
    critic_instruction0 = "Please review the orientation calculations and verify correctness and consistency with geometric principles of the regular 12-gon."
    cot_reflect_desc0 = {
        'instruction': cot_reflect_instruction0,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc0 = {
        'instruction': critic_instruction0,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results0 = await self.reflexion(
        subtask_id="subtask_0",
        cot_reflect_desc=cot_reflect_desc0,
        critic_desc=critic_desc0,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results0['cot_agent'].id}, determining and verifying orientations, thinking: {results0['list_thinking'][0].content}; answer: {results0['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results0['list_feedback']))):
        agents.append(f"Critic agent {results0['critic_agent'].id}, providing feedback, thinking: {results0['list_feedback'][i].content}; answer: {results0['list_correct'][i].content}")
        if i + 1 < len(results0['list_thinking']) and i + 1 < len(results0['list_answer']):
            agents.append(f"Reflexion CoT agent {results0['cot_agent'].id}, refining final answer, thinking: {results0['list_thinking'][i + 1].content}; answer: {results0['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])

    orientation_pairs = []

    for pair_idx, orientation_pair in enumerate(orientation_pairs):
        aggregate_instruction4 = f"Subtask 4: For perpendicular orientation pair {pair_idx+1}, aggregate candidate line segments into two sets for rectangle construction."
        aggregate_desc4 = {
            'instruction': aggregate_instruction4,
            'input': [taskInfo, results0['thinking'], results0['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 0", "answer of subtask 0"]
        }
        results4 = await self.aggregate(
            subtask_id="subtask_4",
            aggregate_desc=aggregate_desc4
        )
        agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, aggregating lines for orientation pair {pair_idx+1}, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])

        cot_sc_instruction3 = f"Subtask 3: Enumerate all possible rectangles by selecting pairs of parallel lines from each orientation set for orientation pair {pair_idx+1}, verifying rectangle validity by checking geometric conditions (parallelism, right angles). Provide multiple solution paths for self-consistency."
        cot_sc_desc3 = {
            'instruction': cot_sc_instruction3,
            'input': [taskInfo, results4['thinking'], results4['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
        }
        results3 = await self.sc_cot(
            subtask_id="subtask_3",
            cot_sc_desc=cot_sc_desc3,
            n_repeat=self.max_sc
        )
        sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        for idx, key in enumerate(results3['list_thinking']):
            agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, enumerating rectangles, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
        logs.append(results3['subtask_desc'])

    debate_instruction7 = "Subtask 7: Review and debate the completeness, uniqueness, and correctness of the rectangle count and enumeration, challenging each other's reasoning paths to ensure robustness."
    final_decision_instruction7 = "Subtask 7: Make final decision on the correctness and completeness of the rectangle enumeration and count."
    debate_desc7 = {
        'instruction': debate_instruction7,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc7 = {
        'instruction': final_decision_instruction7,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results7 = await self.debate(
        subtask_id="subtask_7",
        debate_desc=debate_desc7,
        final_decision_desc=final_decision_desc7,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results7['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating rectangle enumeration, thinking: {results7['list_thinking'][round][idx].content}; answer: {results7['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding on rectangle enumeration correctness, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    cot_instruction8 = "Subtask 8: Format the improved and verified list of rectangles clearly and concisely, enumerating each rectangle explicitly."
    cot_agent_desc8 = {
        'instruction': cot_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
    }
    results8 = await self.cot(
        subtask_id="subtask_8",
        cot_agent_desc=cot_agent_desc8
    )
    agents.append(f"CoT agent {results8['cot_agent'].id}, formatting rectangle list, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    cot_instruction9 = "Subtask 9: Provide only the final count of rectangles as a single integer, without any explanation or additional text."
    cot_agent_desc9 = {
        'instruction': cot_instruction9,
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
    }
    results9 = await self.cot(
        subtask_id="subtask_9",
        cot_agent_desc=cot_agent_desc9
    )
    agents.append(f"CoT agent {results9['cot_agent'].id}, formatting final count, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
    return final_answer, logs
