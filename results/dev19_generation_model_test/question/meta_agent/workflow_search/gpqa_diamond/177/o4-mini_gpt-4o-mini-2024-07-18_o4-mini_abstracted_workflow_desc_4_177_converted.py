async def forward_177(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0, Sub-task 0_1: SC_CoT to determine dimensions of psi and F
    cot_sc_instruction_01 = "Sub-task 0_1: Determine the mass dimensions of the fermion field psi and the gauge field strength F^{mu nu} under hbar=c=1 conventions."
    N1 = self.max_sc
    cot_sc_agents_01 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings_01 = []
    possible_answers_01 = []
    subtask_desc_01 = {"subtask_id":"subtask_0_1","instruction":cot_sc_instruction_01,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking, answer = await cot_sc_agents_01[i]([taskInfo], cot_sc_instruction_01, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_01[i].id}, determining dimensions of psi and F, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_01.append(thinking)
        possible_answers_01.append(answer)
    final_decision_agent_01 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_01 = "Given all the above thinking and answers, find the most consistent and correct mass dimensions for psi and F."
    thinking0_1, answer0_1 = await final_decision_agent_01([taskInfo] + possible_thinkings_01 + possible_answers_01, final_instr_01, is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    subtask_desc_01['response'] = {"thinking":thinking0_1,"answer":answer0_1}
    logs.append(subtask_desc_01)
    print("Step 1: ", sub_tasks[-1])

    # Stage 0, Sub-task 0_2: SC_CoT to confirm sigma is dimensionless
    cot_sc_instruction_02 = "Sub-task 0_2: Confirm that sigma_{mu nu} defined as (i/2)[gamma_mu, gamma_nu] is dimensionless under hbar=c=1 conventions."
    N2 = self.max_sc
    cot_sc_agents_02 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings_02 = []
    possible_answers_02 = []
    subtask_desc_02 = {"subtask_id":"subtask_0_2","instruction":cot_sc_instruction_02,"context":["user query","thinking0_1","answer0_1"],"agent_collaboration":"SC_CoT"}
    for i in range(N2):
        thinking, answer = await cot_sc_agents_02[i]([taskInfo, thinking0_1, answer0_1], cot_sc_instruction_02, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_02[i].id}, confirming dimensionlessness of sigma, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_02.append(thinking)
        possible_answers_02.append(answer)
    final_decision_agent_02 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_02 = "Given all the above thinking and answers, decide the dimension of sigma."
    thinking0_2, answer0_2 = await final_decision_agent_02([taskInfo, thinking0_1, answer0_1] + possible_thinkings_02 + possible_answers_02, final_instr_02, is_sub_task=True)
    sub_tasks.append(f"Sub-task 0_2 output: thinking - {thinking0_2.content}; answer - {answer0_2.content}")
    subtask_desc_02['response'] = {"thinking":thinking0_2,"answer":answer0_2}
    logs.append(subtask_desc_02)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1, Sub-task 1: SC_CoT to compute operator dimension and derive [kappa]
    cot_sc_instruction_1 = "Sub-task 1: Compute the total mass dimension of psi-bar sigma psi F^{mu nu} and derive the mass dimension of kappa assuming L_int has dimension 4."
    N3 = self.max_sc
    cot_sc_agents_1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings_1 = []
    possible_answers_1 = []
    subtask_desc_1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction_1,"context":["user query", "thinking0_1","answer0_1","thinking0_2","answer0_2"],"agent_collaboration":"SC_CoT"}
    for i in range(N3):
        thinking, answer = await cot_sc_agents_1[i]([taskInfo, thinking0_1, answer0_1, thinking0_2, answer0_2], cot_sc_instruction_1, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, computing operator dimension and [kappa], thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings_1.append(thinking)
        possible_answers_1.append(answer)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Given all the above thinking and answers, find the most consistent derivation of [kappa]_M."
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0_1, answer0_1, thinking0_2, answer0_2] + possible_thinkings_1 + possible_answers_1, final_instr_1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc_1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc_1)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2, Sub-task 2: Reflexion to assess renormalizability
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_2 = "Sub-task 2: Assess the renormalizability: decide whether a coupling with the computed mass dimension is renormalizable (requires non-negative dimension)." + reflect_inst
    cot_agent_2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs_2 = [taskInfo, thinking1, answer1]
    subtask_desc_2 = {"subtask_id":"subtask_2","instruction":cot_reflect_instruction_2,"context":["user query","thinking1","answer1"],"agent_collaboration":"Reflexion"}
    thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2.id}, initial renormalizability assessment, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent_2([taskInfo, thinking2, answer2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs_2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_2(cot_inputs_2, cot_reflect_instruction_2, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2.id}, refined assessment, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc_2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc_2)
    print("Step 4: ", sub_tasks[-1])

    # Stage 3, Sub-task 3: Debate to select final MCQ answer
    debate_instr_3 = "Sub-task 3: Select the correct multiple-choice answer matching the computed [kappa]_M and renormalizability conclusion." + "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask_desc_3 = {"subtask_id":"subtask_3","instruction":debate_instr_3,"context":["user query","thinking2","answer2"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3_r, answer3_r = await agent([taskInfo, thinking2, answer2], debate_instr_3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3_r, answer3_r = await agent(inputs, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_r.content}; answer: {answer3_r.content}")
            all_thinking3[r].append(thinking3_r)
            all_answer3[r].append(answer3_r)
    final_decision_agent_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], final_instr_3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc_3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc_3)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs