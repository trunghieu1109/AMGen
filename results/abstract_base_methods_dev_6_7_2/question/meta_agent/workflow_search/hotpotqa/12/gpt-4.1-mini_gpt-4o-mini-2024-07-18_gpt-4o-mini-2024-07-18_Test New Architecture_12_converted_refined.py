async def forward_12(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = "Sub-task 1: Identify the current head of the Foreign Relations Department of the Rastriya Janashakti Party by attempting multiple retrieval strategies including external knowledge retrieval, recent public figures, and leadership roles. Provide partial or approximate identifications if exact information is unavailable."
    critic_instruction1 = "Please review the identification attempts and provide feedback on limitations or missing information."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, identifying head of Foreign Relations Department, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, providing feedback, thinking: {results1['list_feedback'][i].content}; answer: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining identification, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Sub-task 2: List all degree abbreviations (MS, M.S., ScM) and their possible meanings in academic fields relevant to foreign relations and leadership roles."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, listing degree abbreviations and meanings, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    degree_abbreviations = ["MS", "M.S.", "ScM"]
    candidate_fields_all = []

    for idx, degree_abbr in enumerate(degree_abbreviations, start=3):
        cot_sc_instruction = f"Sub-task {idx}: Generate candidate fields of study explicitly associated with the degree abbreviation {degree_abbr} held by the identified person, emphasizing relevance to foreign relations or leadership. Only include fields supported by evidence or common academic usage."
        cot_sc_desc = {
            'instruction': cot_sc_instruction,
            'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], degree_abbr],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", f"degree abbreviation {degree_abbr}"]
        }
        results_sc = await self.sc_cot(
            subtask_id=f"subtask_{idx}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=self.max_sc
        )
        for i in range(self.max_sc):
            agents.append(f"CoT-SC agent {results_sc['cot_agent'][i].id}, candidate fields for {degree_abbr}, thinking: {results_sc['list_thinking'][i]}; answer: {results_sc['list_answer'][i]}")
        sub_tasks.append(f"Sub-task {idx} output: thinking - {results_sc['thinking'].content}; answer - {results_sc['answer'].content}")
        logs.append(results_sc['subtask_desc'])
        candidate_fields_all.append(results_sc['answer'].content)

    cot_sc_instruction6 = "Sub-task 6: Consolidate multiple candidate fields of study generated from Subtasks 3 to 5 into a single coherent field. Only select fields explicitly supported by prior subtasks or external evidence, and apply self-consistency to ensure robustness."
    cot_sc_desc6 = {
        'instruction': cot_sc_instruction6,
        'input': [taskInfo] + candidate_fields_all,
        'temperature': 0.5,
        'context': ["user query", "candidate fields from subtask 3 to subtask 5"]
    }
    results6 = await self.sc_cot(
        subtask_id="subtask_6",
        cot_sc_desc=cot_sc_desc6,
        n_repeat=self.max_sc
    )
    for i in range(self.max_sc):
        agents.append(f"CoT-SC agent {results6['cot_agent'][i].id}, consolidating candidate fields, thinking: {results6['list_thinking'][i]}; answer: {results6['list_answer'][i]}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    cot_reflect_instruction7 = "Sub-task 7: Critically review and validate the consolidated field of study from Subtask 6 against authoritative sources or official data. Cite sources or indicate uncertainty if validation is inconclusive."
    critic_instruction7 = "Please provide feedback on the validation, including any limitations or uncertainties."
    cot_reflect_desc7 = {
        'instruction': cot_reflect_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    critic_desc7 = {
        'instruction': critic_instruction7,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results7 = await self.reflexion(
        subtask_id="subtask_7",
        cot_reflect_desc=cot_reflect_desc7,
        critic_desc=critic_desc7,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, validating consolidated field, thinking: {results7['list_thinking'][0].content}; answer: {results7['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results7['list_feedback']))):
        agents.append(f"Critic agent {results7['critic_agent'].id}, providing feedback, thinking: {results7['list_feedback'][i].content}; answer: {results7['list_correct'][i].content}")
        if i + 1 < len(results7['list_thinking']) and i + 1 < len(results7['list_answer']):
            agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, refining validation, thinking: {results7['list_thinking'][i + 1].content}; answer: {results7['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
