async def forward_159(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    debate_instr1 = "Sub-task 1: Clarify the problem wording and physical parameters: determine precisely whether 'angular distance between the first two minima' means (a) from central maximum to first minimum or (b) between the first and second minima, and confirm that the aperture’s apothem a is the radius (so diameter D=2a). Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max1 = self.max_round
    all_thinking1 = [[] for _ in range(N_max1)]
    all_answer1 = [[] for _ in range(N_max1)]
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": debate_instr1, "context": ["user query"], "agent_collaboration": "Debate"}
    for r in range(N_max1):
        for i, agent in enumerate(debate_agents1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instr1, r, is_sub_task=True)
            else:
                input_infos1 = [taskInfo] + all_thinking1[r-1] + all_answer1[r-1]
                thinking1, answer1 = await agent(input_infos1, debate_instr1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_instr1 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + all_thinking1[-1] + all_answer1[-1], final_instr1, is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction2 = "Sub-task 2: Summarize the diffraction scenario after N→∞: the aperture becomes a circular aperture of radius a, and its Fraunhofer diffraction minima correspond to successive zeros of the Bessel function J1. Explicitly note to avoid invoking 1.22 λ/D for first vs. successive minima."
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["user query", "response of subtask_1"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents2:
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent answer for summarizing the diffraction scenario.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_instruction3 = "Sub-task 3: Look up or compute accurately the first two positive zeros of J1: j1,1 ≈ 3.8317 and j1,2 ≈ 7.0156. Ensure numerical precision to at least four significant figures."
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "context": ["user query", "response of subtask_2"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents3:
        thinking3, answer3 = await agent([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_thinkings3.append(thinking3)
        possible_answers3.append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, "Sub-task 3: Synthesize and choose the most consistent numerical values for j1,1 and j1,2.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_instruction4 = "Sub-task 4: Derive the formula for angular separation Δθ = (j1,2 – j1,1)·(λ/(2π·a)), being careful to use a as the radius. Then perform the explicit numeric division: compute (7.0156–3.8317)/(2π) ≈ 0.507, and multiply by λ/a. Include an inline check that 3.1839/(2π) ≈ 0.507 to avoid the prior algebraic slip that yielded 1.220."
    cot_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction4, "context": ["user query", "response of subtask_3"], "agent_collaboration": "SC_CoT"}
    for agent in cot_agents4:
        thinking4, answer4 = await agent([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_thinkings4.append(thinking4)
        possible_answers4.append(answer4)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4, "Sub-task 4: Synthesize and choose the correct derived formula and numeric result for Δθ.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    reflect_inst5 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction5 = "Sub-task 5: Verify the physical correctness of the derived Δθ by cross-checking with the standard Airy-pattern formula: θ1 ≃ 1.22 λ/D for the first minimum from central maximum, and confirm that our computed Δθ between the first two minima (≈0.507 λ/a) is consistent with that context and uses D=2a properly." + reflect_inst5
    cot_agent5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs5 = [taskInfo, thinking4, answer4]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": cot_reflect_instruction5, "context": ["user query", "response of subtask_4"], "agent_collaboration": "Reflexion"}
    thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    critic_inst5 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], critic_inst5, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback5.content}; correct: {correct5.content}")
        if correct5.content == "True":
            break
        cot_inputs5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    debate_instr6 = "Sub-task 6: Compare the final numerical result Δθ ≈ 0.507 λ/a against the provided answer choices (1.220, 0.610, 0.500, 0.506 λ/a) and select the best match. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max6)]
    all_answer6 = [[] for _ in range(N_max6)]
    subtask_desc6 = {"subtask_id": "subtask_6", "instruction": debate_instr6, "context": ["user query", "response of subtask_5"], "agent_collaboration": "Debate"}
    for r in range(N_max6):
        for i, agent in enumerate(debate_agents6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instr6, r, is_sub_task=True)
            else:
                input_infos6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos6, debate_instr6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_instr6 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent6([taskInfo, thinking5, answer5] + all_thinking6[-1] + all_answer6[-1], final_instr6, is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs