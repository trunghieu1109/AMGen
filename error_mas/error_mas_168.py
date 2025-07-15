async def forward_168(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Extract and summarize defining features (Debate)
    debate_instr_0 = "Sub-task 1: Extract and summarize the defining features of the original and variant decay processes, including particle types, decay products, and known spectral properties. Ensure clarity on the masses and nature of particles (noting M is massless), and the known continuity and endpoint Q of the original E particle spectrum." + \
                   " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_0 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_0 = self.max_round
    all_thinking_0 = [[] for _ in range(N_max_0)]
    all_answer_0 = [[] for _ in range(N_max_0)]
    subtask_desc_0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": debate_instr_0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_0):
        for i, agent in enumerate(debate_agents_0):
            if r == 0:
                thinking_0, answer_0 = await agent([taskInfo], debate_instr_0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_0[r-1] + all_answer_0[r-1]
                thinking_0, answer_0 = await agent(input_infos_0, debate_instr_0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_0.content}; answer: {answer_0.content}")
            all_thinking_0[r].append(thinking_0)
            all_answer_0[r].append(answer_0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_0 = "Sub-task 1: Extract and summarize decay features. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_0, answer_0 = await final_decision_agent_0([taskInfo] + all_thinking_0[-1] + all_answer_0[-1], final_instr_0, is_sub_task=True)
    agents.append(f"Final Decision agent stage_0, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    # Stage 1 Subtask 1: Formulate explicit relativistic energy and momentum conservation equations (SC_CoT)
    cot_sc_instruction_1 = "Sub-task 1: Formulate explicit relativistic energy and momentum conservation equations for both the original decay (2A -> 2B + 2E + 2V) and the variant decay (2A -> 2B + 2E + M). Write symbolic expressions for the endpoint energies Q_old and Q_new, including rest mass terms, to rigorously analyze how replacing two massive V particles with one massless M affects the maximum available kinetic energy for the E particles. Use stepwise, physics-grounded reasoning."
    N_sc = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_1 = []
    possible_thinkings_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking_1, answer_1 = await cot_agents_1[i]([taskInfo, thinking_0, answer_0], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, thinking: {thinking_1.content}; answer: {answer_1.content}")
        possible_answers_1.append(answer_1)
        possible_thinkings_1.append(thinking_1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Sub-task 1: Synthesize and choose the most consistent and correct solutions for the energy conservation and endpoint expressions."
    thinking_1, answer_1 = await final_decision_agent_1([taskInfo, thinking_0, answer_0] + possible_thinkings_1 + possible_answers_1, final_instr_1, is_sub_task=True)
    agents.append(f"Final Decision agent stage_1.subtask_1, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    # Stage 1 Subtask 2: Analyze impact on shape and continuity of spectrum (Debate)
    debate_instr_1_2 = "Sub-task 2: Analyze the impact of the altered final state on the shape and continuity of the total energy spectrum of the E particles, incorporating the results from the explicit energy-balance derivation. Emphasize that the spectrum remains continuous due to multi-body kinematics, but the shape and endpoint shift according to the energy available." + \
                     " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_2 = self.max_round
    all_thinking_1_2 = [[] for _ in range(N_max_1_2)]
    all_answer_1_2 = [[] for _ in range(N_max_1_2)]
    subtask_desc_1_2 = {
        "subtask_id": "stage_1.subtask_2",
        "instruction": debate_instr_1_2,
        "context": ["user query", thinking_0, answer_0, thinking_1, answer_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_2):
        for i, agent in enumerate(debate_agents_1_2):
            if r == 0:
                thinking_1_2, answer_1_2 = await agent([taskInfo, thinking_0, answer_0, thinking_1, answer_1], debate_instr_1_2, r, is_sub_task=True)
            else:
                input_infos_1_2 = [taskInfo, thinking_0, answer_0, thinking_1, answer_1] + all_thinking_1_2[r-1] + all_answer_1_2[r-1]
                thinking_1_2, answer_1_2 = await agent(input_infos_1_2, debate_instr_1_2, r, is_sub_task=True)
            agents.append(f"Debate agent stage_1.subtask_2 {agent.id}, round {r}, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
            all_thinking_1_2[r].append(thinking_1_2)
            all_answer_1_2[r].append(answer_1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1_2 = "Sub-task 2: Analyze spectrum shape and continuity. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_1_2, answer_1_2 = await final_decision_agent_1_2([taskInfo, thinking_0, answer_0, thinking_1, answer_1] + all_thinking_1_2[-1] + all_answer_1_2[-1], final_instr_1_2, is_sub_task=True)
    agents.append(f"Final Decision agent stage_1.subtask_2, thinking: {thinking_1_2.content}; answer: {answer_1_2.content}")
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking_1_2.content}; answer - {answer_1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking_1_2, "answer": answer_1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    # Stage 2 Subtask 1: Synthesize findings and classify spectrum (Debate)
    debate_instr_2 = "Sub-task 1: Synthesize the findings from the kinematic derivation and spectral shape analysis to classify the resulting energy spectrum of the E particles in the variant decay. Confirm that the spectrum remains continuous with an adjusted shape and that the endpoint increases compared to the original decay." + \
                   " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_max_2)]
    all_answer_2 = [[] for _ in range(N_max_2)]
    subtask_desc_2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_2,
        "context": ["user query", thinking_1, answer_1, thinking_1_2, answer_1_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking_2, answer_2 = await agent([taskInfo, thinking_1, answer_1, thinking_1_2, answer_1_2], debate_instr_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking_1, answer_1, thinking_1_2, answer_1_2] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking_2, answer_2 = await agent(input_infos_2, debate_instr_2, r, is_sub_task=True)
            agents.append(f"Debate agent stage_2.subtask_1 {agent.id}, round {r}, thinking: {thinking_2.content}; answer: {answer_2.content}")
            all_thinking_2[r].append(thinking_2)
            all_answer_2[r].append(answer_2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Sub-task 1: Synthesize and finalize classification of the energy spectrum. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking_2, answer_2 = await final_decision_agent_2([taskInfo, thinking_1, answer_1, thinking_1_2, answer_1_2] + all_thinking_2[-1] + all_answer_2[-1], final_instr_2, is_sub_task=True)
    agents.append(f"Final Decision agent stage_2.subtask_1, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2, answer_2, sub_tasks, agents)
    return final_answer, logs
