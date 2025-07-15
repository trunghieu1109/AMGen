async def forward_151(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Biological Context Clarification (Debate)
    debate_instr_0_1 = (
        "Sub-task 1: Summarize and clarify the biological context: the role of the quorum-sensing peptide, "
        "shmoo formation in yeast, and the significance of active chromatin in this process. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) 
                         for role in self.debate_role]
    all_thinking_0_1 = []
    all_answer_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_0_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_0_1):
        thinking, answer = await agent([taskInfo], debate_instr_0_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, subtask_1, thinking: {thinking.content}; answer: {answer.content}")
        all_thinking_0_1.append(thinking)
        all_answer_0_1.append(answer)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await final_decision_agent_0_1(
        [taskInfo] + all_thinking_0_1 + all_answer_0_1,
        "Sub-task 1: Summarize and clarify biological context. Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, subtask_1, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking_0_1, "answer": answer_0_1}
    logs.append(subtask_desc_0_1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 0: Analyze protein complexes (SC_CoT)
    cot_sc_instruction_0_2 = (
        "Sub-task 2: Based on the output from Sub-task 1, analyze the functions and chromatin association of the four protein complexes "
        "(pre-initiation complex, pre-replication complex, enhancer protein complex, nucleosome histone complex) in the context of active chromatin and shmoo formation."
    )
    N_sc = self.max_sc
    cot_agents_0_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers_0_2 = []
    possible_thinkings_0_2 = []
    subtask_desc_0_2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_0_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking, answer = await cot_agents_0_2[i]([taskInfo, thinking_0_1, answer_0_1], cot_sc_instruction_0_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0_2[i].id}, subtask_2, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_0_2.append(answer)
        possible_thinkings_0_2.append(thinking)
    final_decision_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await final_decision_agent_0_2(
        [taskInfo, thinking_0_1, answer_0_1] + possible_thinkings_0_2 + possible_answers_0_2,
        "Sub-task 2: Synthesize and choose the most consistent and correct analysis of protein complexes.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, subtask_2, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    subtask_desc_0_2['response'] = {"thinking": thinking_0_2, "answer": answer_0_2}
    logs.append(subtask_desc_0_2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 0: Generate hypotheses (Debate)
    debate_instr_0_3 = (
        "Sub-task 3: Generate variant hypotheses on which complexes are expected to be enriched or depleted in active chromatin during shmoo formation, "
        "considering the experimental method (ChIP-MS) and biological context. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_0_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) 
                         for role in self.debate_role]
    all_thinking_0_3 = []
    all_answer_0_3 = []
    subtask_desc_0_3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_0_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Debate"
    }
    for i, agent in enumerate(debate_agents_0_3):
        thinking, answer = await agent([taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2], debate_instr_0_3, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, subtask_3, thinking: {thinking.content}; answer: {answer.content}")
        all_thinking_0_3.append(thinking)
        all_answer_0_3.append(answer)
    final_decision_agent_0_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_0_3, answer_0_3 = await final_decision_agent_0_3(
        [taskInfo, thinking_0_1, answer_0_1, thinking_0_2, answer_0_2] + all_thinking_0_3 + all_answer_0_3,
        "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, subtask_3, thinking: {thinking_0_3.content}; answer: {answer_0_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_0_3.content}; answer - {answer_0_3.content}")
    subtask_desc_0_3['response'] = {"thinking": thinking_0_3, "answer": answer_0_3}
    logs.append(subtask_desc_0_3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 1: Identify least observed complex (SC_CoT)
    cot_sc_instruction_1_1 = (
        "Sub-task 1: Apply reasoning to identify which protein complex will be least observed in the ChIP-MS assay of active chromatin in the shmoo, "
        "based on the generated hypotheses and biological knowledge from previous subtasks."
    )
    N_sc_1 = self.max_sc
    cot_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1)]
    possible_answers_1_1 = []
    possible_thinkings_1_1 = []
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction_1_1,
        "context": ["user query", "thinking of stage_0.subtask_3", "answer of stage_0.subtask_3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1):
        thinking, answer = await cot_agents_1_1[i]([taskInfo, thinking_0_3, answer_0_3], cot_sc_instruction_1_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_1[i].id}, subtask_1 stage_1, thinking: {thinking.content}; answer: {answer.content}")
        possible_answers_1_1.append(answer)
        possible_thinkings_1_1.append(thinking)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1_1, answer_1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking_0_3, answer_0_3] + possible_thinkings_1_1 + possible_answers_1_1,
        "Sub-task 1 stage 1: Synthesize and choose the most consistent and correct identification of the least observed protein complex.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, subtask_1 stage_1, thinking: {thinking_1_1.content}; answer: {answer_1_1.content}")
    sub_tasks.append(f"Sub-task 1 stage 1 output: thinking - {thinking_1_1.content}; answer - {answer_1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking_1_1, "answer": answer_1_1}
    logs.append(subtask_desc_1_1)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_1_1, answer_1_1, sub_tasks, agents)
    return final_answer, logs
