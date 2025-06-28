async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    molecules = [
        "quinuclidine",
        "triisopropyl borate",
        "benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone",
        "triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone"
    ]
    stage_1_subtask_1_instruction = (
        "Sub-task 1: Collect and clearly define the 3D molecular structures and geometries of each given molecule: "
        "quinuclidine, triisopropyl borate, benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone, and triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone. "
        "Use authoritative sources or computational optimization data to provide detailed 3D structural descriptions enabling accurate symmetry analysis."
    )
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": stage_1_subtask_1_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], stage_1_subtask_1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, collecting 3D molecular structures, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    stage_1_subtask_2_instruction = (
        "Sub-task 2: Enumerate exhaustively all symmetry elements present in each molecule's 3D structure, including identity (E), rotational axes (C3, C2, etc.), mirror planes (σh, σv), inversion centers (i), and improper rotation axes (S3, etc.). "
        "Use authoritative point group definitions as reference and the structural data from Sub-task 1."
    )
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": stage_1_subtask_2_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], stage_1_subtask_2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, enumerating symmetry elements, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[answer2_content]
    answer2 = answermapping_2[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    stage_2_subtask_3_instruction = (
        "Sub-task 3: Assign the full point group symmetry to each molecule based on the complete set of identified symmetry elements. "
        "Explicitly distinguish C3h from higher symmetry groups such as D3h by verifying the absence of additional symmetry elements beyond those defining C3h (E, C3, C3^2, σh). "
        "Use outputs from Sub-task 2 and authoritative point group definitions."
    )
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": stage_2_subtask_3_instruction,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_3 = [taskInfo, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, stage_2_subtask_3_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, assigning point groups, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3], "Please review the point group assignment and verify the absence of additional symmetry elements beyond C3h, especially to exclude higher symmetry groups like D3h.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, stage_2_subtask_3_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining point group assignment, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    stage_2_subtask_4_instruction = (
        "Sub-task 4: Cross-verify the assigned point groups against authoritative character tables and symmetry group definitions to confirm that molecules assigned as C3h possess exactly the symmetry elements of C3h and no additional elements indicative of higher symmetry groups. "
        "Use outputs from Sub-task 3 and authoritative references."
    )
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": stage_2_subtask_4_instruction,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, stage_2_subtask_4_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, cross-verifying point groups, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4], "Please review the cross-verification of assigned point groups against authoritative tables and confirm exact C3h membership without higher symmetry.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, stage_2_subtask_4_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining cross-verification, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    stage_2_subtask_5_instruction = (
        "Sub-task 5: Evaluate and confirm the structural assumptions critical for C3h symmetry, such as molecular planarity and the presence of a horizontal mirror plane (σh), "
        "using computational or experimental structural data where available to reduce uncertainty. Use outputs from Sub-tasks 1 and 4."
    )
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": stage_2_subtask_5_instruction,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Reflexion"
    }
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_5 = [taskInfo, thinking1, answer1, thinking4, answer4]
    thinking5, answer5 = await cot_agent_5(cot_inputs_5, stage_2_subtask_5_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_5.id}, confirming structural assumptions, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent_5([taskInfo, thinking5, answer5], "Please review the evaluation of structural assumptions critical for C3h symmetry, including planarity and σh presence, and provide limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_5.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_5.extend([thinking5, answer5, feedback])
        thinking5, answer5 = await cot_agent_5(cot_inputs_5, stage_2_subtask_5_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_5.id}, refining structural assumption evaluation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    stage_2_subtask_6_instruction = (
        "Sub-task 6: Compare the verified point group symmetries of all four molecules and select the molecule(s) that exhibit exactly C3h symmetry, "
        "explicitly excluding those with higher symmetry groups that contain C3h as a subgroup. Use outputs from Sub-tasks 4 and 5."
    )
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": stage_2_subtask_6_instruction,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "Debate"
    }
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking4, answer4, thinking5, answer5], stage_2_subtask_6_instruction, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking4, answer4, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, stage_2_subtask_6_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, selecting molecule(s) with exact C3h symmetry, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on which molecule(s) exhibit exactly C3h symmetry.", is_sub_task=True)
    agents.append(f"Final Decision agent, determining molecule(s) with exact C3h symmetry, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6["response"] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs