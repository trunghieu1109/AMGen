async def forward_151(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = (
        "Sub-task 1: Analyze and classify the biological elements involved: quorum-sensing peptide, shmoo formation in yeast, "
        "and the four protein complexes (pre-initiation, pre-replication, enhancer, nucleosome histone) with respect to their roles and characteristics."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_0 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_0, answer_0 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, analyzing biological elements, thinking: {thinking_0.content}; answer: {answer_0.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0.content}; answer - {answer_0.content}")
    subtask_desc_0['response'] = {"thinking": thinking_0, "answer": answer_0}
    logs.append(subtask_desc_0)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_1 = (
        "Sub-task 1: Evaluate the conformity and validity of the association between the observed shmoo formation and active chromatin states, "
        "focusing on which protein complexes are expected to be present or absent in active chromatin during transcriptional activation. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]
    subtask_desc_1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_1,
        "context": ["user query", thinking_0, answer_0],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking_1, answer_1 = await agent([taskInfo, thinking_0, answer_0], debate_instruction_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo, thinking_0, answer_0] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking_1, answer_1 = await agent(input_infos_1, debate_instruction_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating active chromatin association, thinking: {thinking_1.content}; answer: {answer_1.content}")
            all_thinking_1[r].append(thinking_1)
            all_answer_1[r].append(answer_1)
    final_decision_instruction_1 = (
        "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_1, answer_1 = await final_decision_agent_1(
        [taskInfo, thinking_0, answer_0] + all_thinking_1[-1] + all_answer_1[-1],
        "Sub-task 1: Evaluate active chromatin association." + final_decision_instruction_1,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, evaluating active chromatin association, thinking: {thinking_1.content}; answer: {answer_1.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_1.content}; answer - {answer_1.content}")
    subtask_desc_1['response'] = {"thinking": thinking_1, "answer": answer_1}
    logs.append(subtask_desc_1)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction_2 = (
        "Sub-task 1: Analyze and classify the protein complexes based on their likelihood of being detected by chromatin immunoprecipitation followed by mass spectrometry of active chromatin in the shmoo, "
        "and identify the least represented complex in this assay. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_2 = self.max_round
    all_thinking_2 = [[] for _ in range(N_max_2)]
    all_answer_2 = [[] for _ in range(N_max_2)]
    subtask_desc_2 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking_0, answer_0, thinking_1, answer_1],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking_2, answer_2 = await agent([taskInfo, thinking_0, answer_0, thinking_1, answer_1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking_0, answer_0, thinking_1, answer_1] + all_thinking_2[r-1] + all_answer_2[r-1]
                thinking_2, answer_2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing protein complex detection, thinking: {thinking_2.content}; answer: {answer_2.content}")
            all_thinking_2[r].append(thinking_2)
            all_answer_2[r].append(answer_2)
    final_decision_instruction_2 = (
        "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2, answer_2 = await final_decision_agent_2(
        [taskInfo, thinking_0, answer_0, thinking_1, answer_1] + all_thinking_2[-1] + all_answer_2[-1],
        "Sub-task 1: Identify least represented protein complex in ChIP-MS assay." + final_decision_instruction_2,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, identifying least represented complex, thinking: {thinking_2.content}; answer: {answer_2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_2.content}; answer - {answer_2.content}")
    subtask_desc_2['response'] = {"thinking": thinking_2, "answer": answer_2}
    logs.append(subtask_desc_2)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2, answer_2, sub_tasks, agents)
    return final_answer, logs
