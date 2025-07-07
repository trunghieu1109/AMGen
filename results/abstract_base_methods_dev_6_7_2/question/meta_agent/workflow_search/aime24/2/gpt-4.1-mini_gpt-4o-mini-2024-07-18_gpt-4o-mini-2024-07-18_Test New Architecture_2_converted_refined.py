async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Sub-task 1: Define the sample space including vertices, color assignments, and rotation group for a regular octagon with context from taskInfo"
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, defining sample space, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_reflect_instruction2 = "Sub-task 2: Critically review and correct assumptions about the constraints on blue and red vertices under rotation; verify that the number of blue vertices must be less than or equal to the number of red vertices and that the rotation maps the blue set into a subset of the red set, not necessarily a perfect matching, with context from taskInfo and output of Sub-task 1"
    critic_instruction2 = "Please review the assumptions about blue and red vertex mappings under rotation and provide feedback on correctness and limitations."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, reviewing constraints on blue and red vertices, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, providing feedback, thinking: {results2['list_feedback'][i].content}; answer: {results2['list_correct'][i].content}")
        if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
            agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining answer, thinking: {results2['list_thinking'][i + 1].content}; answer: {results2['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    cot_sc_instruction3 = "Sub-task 3: Explore multiple reasoning paths to characterize favorable colorings using group action and Burnside's lemma, with context from taskInfo and outputs of Sub-tasks 1 and 2"
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, characterizing favorable colorings, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    favorable_counts = []
    for rotation in range(8):
        cot_sc_instruction_loop = f"Sub-task {4+rotation}: For rotation {rotation}, use cycle decomposition to count colorings fixed by this rotation under the problem's constraints, verifying all possible configurations with context from taskInfo and output of Sub-task 3"
        cot_sc_desc_loop = {
            'instruction': cot_sc_instruction_loop,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        results_loop = await self.sc_cot(
            subtask_id=f"subtask_{4+rotation}",
            cot_sc_desc=cot_sc_desc_loop,
            n_repeat=self.max_sc
        )
        for idx, key in enumerate(results_loop['list_thinking']):
            agents.append(f"CoT-SC agent {results_loop['cot_agent'][idx].id}, counting fixed colorings for rotation {rotation}, thinking: {results_loop['list_thinking'][idx]}; answer: {results_loop['list_answer'][idx]}")
        sub_tasks.append(f"Sub-task {4+rotation} output: thinking - {results_loop['thinking'].content}; answer - {results_loop['answer'].content}")
        logs.append(results_loop['subtask_desc'])
        favorable_counts.append(results_loop['answer'].content)
    aggregate_instruction = "Sub-task 12: Aggregate counts over all rotations applying Burnside's lemma explicitly to compute total favorable colorings, verifying the aggregated count mathematically with context from taskInfo and counts from Sub-tasks 4 to 11"
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + favorable_counts,
        'temperature': 0.5,
        'context': ["user query", "counts from subtasks 4 to 11"]
    }
    results_aggregate = await self.sc_cot(
        subtask_id="subtask_12",
        cot_sc_desc=aggregate_desc,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results_aggregate['list_thinking']):
        agents.append(f"CoT-SC agent {results_aggregate['cot_agent'][idx].id}, aggregating counts with Burnside's lemma, thinking: {results_aggregate['list_thinking'][idx]}; answer: {results_aggregate['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    cot_instruction13 = "Sub-task 13: Implement the counting algorithm step-by-step in code to count favorable colorings using Burnside's lemma with context from taskInfo and output of Sub-task 12"
    cot_agent_desc13 = {
        'instruction': cot_instruction13,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 12", "answer of subtask 12"],
        'entry_point': "count_favorable_colorings"
    }
    results13 = await self.cot(
        subtask_id="subtask_13",
        cot_agent_desc=cot_agent_desc13
    )
    agents.append(f"CoT agent {results13['cot_agent'].id}, implementing counting code, thinking: {results13['thinking'].content}; answer: {results13['answer'].content}")
    sub_tasks.append(f"Sub-task 13 output: thinking - {results13['thinking'].content}; answer - {results13['answer'].content}")
    logs.append(results13['subtask_desc'])
    cot_reflect_instruction14 = "Sub-task 14: Verify and format the final answer ensuring consistency with validated computations and format output strictly as required with context from taskInfo and outputs of Sub-tasks 12 and 13"
    critic_instruction14 = "Please review the final answer formatting and correctness, providing feedback and corrections if needed."
    cot_reflect_desc14 = {
        'instruction': cot_reflect_instruction14,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer'], results13['thinking'], results13['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 12", "answer of subtask 12", "thinking of subtask 13", "answer of subtask 13"]
    }
    critic_desc14 = {
        'instruction': critic_instruction14,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results14 = await self.reflexion(
        subtask_id="subtask_14",
        cot_reflect_desc=cot_reflect_desc14,
        critic_desc=critic_desc14,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results14['cot_agent'].id}, verifying final answer, thinking: {results14['list_thinking'][0].content}; answer: {results14['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results14['list_feedback']))):
        agents.append(f"Critic agent {results14['critic_agent'].id}, providing feedback, thinking: {results14['list_feedback'][i].content}; answer: {results14['list_correct'][i].content}")
        if i + 1 < len(results14['list_thinking']) and i + 1 < len(results14['list_answer']):
            agents.append(f"Reflexion CoT agent {results14['cot_agent'].id}, refining final answer, thinking: {results14['list_thinking'][i + 1].content}; answer: {results14['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 14 output: thinking - {results14['thinking'].content}; answer - {results14['answer'].content}")
    logs.append(results14['subtask_desc'])
    cot_sc_instruction16 = "Sub-task 16: Validate the final probability fraction for correctness and reduced form with context from taskInfo and output of Sub-task 14"
    cot_sc_desc16 = {
        'instruction': cot_sc_instruction16,
        'input': [taskInfo, results14['thinking'], results14['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 14", "answer of subtask 14"]
    }
    results16 = await self.sc_cot(
        subtask_id="subtask_16",
        cot_sc_desc=cot_sc_desc16,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results16['list_thinking']):
        agents.append(f"CoT-SC agent {results16['cot_agent'][idx].id}, validating probability fraction, thinking: {results16['list_thinking'][idx]}; answer: {results16['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 16 output: thinking - {results16['thinking'].content}; answer - {results16['answer'].content}")
    logs.append(results16['subtask_desc'])
    cot_sc_instruction17 = "Sub-task 17: Enhance clarity of the probability derivation and simplify presentation with context from taskInfo and output of Sub-task 16"
    cot_sc_desc17 = {
        'instruction': cot_sc_instruction17,
        'input': [taskInfo, results16['thinking'], results16['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 16", "answer of subtask 16"]
    }
    results17 = await self.sc_cot(
        subtask_id="subtask_17",
        cot_sc_desc=cot_sc_desc17,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results17['list_thinking']):
        agents.append(f"CoT-SC agent {results17['cot_agent'][idx].id}, enhancing clarity, thinking: {results17['list_thinking'][idx]}; answer: {results17['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 17 output: thinking - {results17['thinking'].content}; answer - {results17['answer'].content}")
    logs.append(results17['subtask_desc'])
    cot_sc_instruction18 = "Sub-task 18: Format the final answer as m+n after reduction with context from taskInfo and output of Sub-task 17"
    cot_sc_desc18 = {
        'instruction': cot_sc_instruction18,
        'input': [taskInfo, results17['thinking'], results17['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 17", "answer of subtask 17"]
    }
    results18 = await self.sc_cot(
        subtask_id="subtask_18",
        cot_sc_desc=cot_sc_desc18,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results18['list_thinking']):
        agents.append(f"CoT-SC agent {results18['cot_agent'][idx].id}, formatting final answer, thinking: {results18['list_thinking'][idx]}; answer: {results18['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 18 output: thinking - {results18['thinking'].content}; answer - {results18['answer'].content}")
    logs.append(results18['subtask_desc'])
    final_answer = await self.make_final_answer(results18['thinking'], results18['answer'], sub_tasks, agents)
    return final_answer, logs
