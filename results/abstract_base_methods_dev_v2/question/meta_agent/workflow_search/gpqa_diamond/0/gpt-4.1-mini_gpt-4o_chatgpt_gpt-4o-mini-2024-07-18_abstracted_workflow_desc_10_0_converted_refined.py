async def forward_0(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    molecules = [
        ("triisopropyl borate", "subtask_1"),
        ("quinuclidine", "subtask_2"),
        ("benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone", "subtask_3"),
        ("triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone", "subtask_4")
    ]
    stage1_results = {}
    for mol_name, subtask_id in molecules:
        cot_sc_instruction = f"Sub-task {subtask_id[-1]}: Retrieve detailed 3D molecular geometry and explicit structural diagrams or coordinate data for {mol_name}, including all relevant atoms and bonds to clarify symmetry elements."
        N = self.max_sc
        cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
        possible_answers = []
        thinkingmapping = {}
        answermapping = {}
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": cot_sc_instruction,
            "context": ["user query"],
            "agent_collaboration": "SC_CoT"
        }
        for i in range(N):
            thinking, answer = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents[i].id}, retrieving 3D structure of {mol_name}, thinking: {thinking.content}; answer: {answer.content}")
            possible_answers.append(answer.content)
            thinkingmapping[answer.content] = thinking
            answermapping[answer.content] = answer
        answer_content = Counter(possible_answers).most_common(1)[0][0]
        thinking = thinkingmapping[answer_content]
        answer = answermapping[answer_content]
        sub_tasks.append(f"Sub-task {subtask_id} output: thinking - {thinking.content}; answer - {answer.content}")
        subtask_desc['response'] = {"thinking": thinking, "answer": answer}
        logs.append(subtask_desc)
        stage1_results[subtask_id] = (thinking, answer)
    stage2_results = {}
    symmetry_summary = {}
    for mol_name, subtask_id, dep_subtask in [
        ("triisopropyl borate", "subtask_5", "subtask_1"),
        ("quinuclidine", "subtask_6", "subtask_2"),
        ("benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone", "subtask_7", "subtask_3"),
        ("triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone", "subtask_8", "subtask_4")
    ]:
        thinking1, answer1 = stage1_results[dep_subtask]
        cot_sc_instruction = f"Sub-task {subtask_id[-1]}: Analyze the detailed 3D structure of {mol_name} to explicitly identify and justify the presence or absence of all key symmetry elements (C3 axis, horizontal mirror plane σh, vertical mirror planes σv, etc.) based on the molecular geometry. Provide clear reasoning referencing the 3D structure."
        N = self.max_sc
        cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
        possible_answers = []
        thinkingmapping = {}
        answermapping = {}
        subtask_desc = {
            "subtask_id": subtask_id,
            "instruction": cot_sc_instruction,
            "context": ["user query", f"thinking of {dep_subtask}", f"answer of {dep_subtask}"],
            "agent_collaboration": "SC_CoT"
        }
        for i in range(N):
            thinking, answer = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents[i].id}, analyzing symmetry elements of {mol_name}, thinking: {thinking.content}; answer: {answer.content}")
            possible_answers.append(answer.content)
            thinkingmapping[answer.content] = thinking
            answermapping[answer.content] = answer
        answer_content = Counter(possible_answers).most_common(1)[0][0]
        thinking = thinkingmapping[answer_content]
        answer = answermapping[answer_content]
        sub_tasks.append(f"Sub-task {subtask_id} output: thinking - {thinking.content}; answer - {answer.content}")
        subtask_desc['response'] = {"thinking": thinking, "answer": answer}
        logs.append(subtask_desc)
        stage2_results[subtask_id] = (thinking, answer)
        symmetry_summary[mol_name] = answer.content
    cot_reflect_instruction = "Sub-task 9: Review all identified symmetry elements for each molecule from previous analyses. Critically evaluate any claims of missing or present symmetry elements, ensuring all conclusions are justified by the 3D structures. If inconsistencies or unsupported claims are found, re-analyze and refine the symmetry identifications accordingly."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo]
    for k in sorted(stage2_results.keys()):
        cot_inputs.append(stage2_results[k][0])
        cot_inputs.append(stage2_results[k][1])
    subtask_desc9 = {
        "subtask_id": "subtask_9",
        "instruction": cot_reflect_instruction,
        "context": ["user query"] + [f"thinking of {k}" for k in stage2_results.keys()] + [f"answer of {k}" for k in stage2_results.keys()],
        "agent_collaboration": "Reflexion"
    }
    thinking9, answer9 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, reviewing symmetry summaries, thinking: {thinking9.content}; answer: {answer9.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking9, answer9], "Please review the symmetry element identifications and provide any limitations or errors.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content.strip().lower() == "true":
            break
        cot_inputs.extend([thinking9, answer9, feedback])
        thinking9, answer9 = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refining symmetry review, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    subtask_desc9['response'] = {"thinking": thinking9, "answer": answer9}
    logs.append(subtask_desc9)
    debate_instruction = "Sub-task 10: Based on the reviewed symmetry analyses, debate and select which molecule among triisopropyl borate, quinuclidine, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone exhibits C3h symmetry. Provide the answer in a clear multiple-choice format referencing the molecule name and choice number."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc10 = {
        "subtask_id": "subtask_10",
        "instruction": debate_instruction,
        "context": ["user query"] + [f"thinking of {k}" for k in stage2_results.keys()] + [f"answer of {k}" for k in stage2_results.keys()] + ["thinking of subtask 9", "answer of subtask 9"],
        "agent_collaboration": "Debate"
    }
    thinking_inputs = [taskInfo]
    for k in sorted(stage2_results.keys()):
        thinking_inputs.append(stage2_results[k][0])
        thinking_inputs.append(stage2_results[k][1])
    thinking_inputs.append(thinking9)
    thinking_inputs.append(answer9)
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking, answer = await agent(thinking_inputs, debate_instruction, r, is_sub_task=True)
            else:
                input_infos = thinking_inputs + all_thinking[r-1] + all_answer[r-1]
                thinking, answer = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, debating molecule with C3h symmetry, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_final, answer_final = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], "Sub-task 10: Make final decision on which molecule has C3h symmetry in multiple-choice format.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding molecule with C3h symmetry, thinking: {thinking_final.content}; answer: {answer_final.content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {thinking_final.content}; answer - {answer_final.content}")
    subtask_desc10['response'] = {"thinking": thinking_final, "answer": answer_final}
    logs.append(subtask_desc10)
    final_answer = await self.make_final_answer(thinking_final, answer_final, sub_tasks, agents)
    return final_answer, logs