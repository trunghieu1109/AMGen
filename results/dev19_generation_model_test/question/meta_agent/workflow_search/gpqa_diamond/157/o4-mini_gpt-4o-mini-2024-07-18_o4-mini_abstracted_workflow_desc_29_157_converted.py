async def forward_157(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0, Sub-task 1: SC_CoT
    sc_instruction = "Sub-task 1: Extract and summarize the transcription factor’s domain structure, activation steps, and the nature of mutations X and Y."
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc0 = {"subtask_id": "subtask_1", "instruction": sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking1_i, answer1_i = await sc_agents[i]([taskInfo], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents[i].id}, extracting summary, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings.append(thinking1_i)
        possible_answers.append(answer1_i)
    final_decision_agent0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent0(
        [taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 1: Synthesize and choose the most consistent summary.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc0['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1, Sub-task 2: Debate
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = (
        "Sub-task 2: Determine which dominant-negative mechanism best fits a heterozygous dimerization-domain mutant that inactivates the WT allele." 
        + debate_instr
    )
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds = self.max_round
    all_thinking2 = [[] for _ in range(rounds)]
    all_answer2 = [[] for _ in range(rounds)]
    subtask_desc1 = {"subtask_id": "subtask_2", "instruction": debate_instruction, "context": ["user query", "thinking1", "answer1"], "agent_collaboration": "Debate"}
    for r in range(rounds):
        for agent in debate_agents:
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instruction, r, is_sub_task=True)
            else:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent1(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc1['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2, Sub-task 3: Debate
    debate_instr2 = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction2 = (
        "Sub-task 3: Evaluate and prioritize the answer choices against the inferred mechanism to select the most likely observed phenotype." 
        + debate_instr2
    )
    debate_agents2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(rounds)]
    all_answer3 = [[] for _ in range(rounds)]
    subtask_desc2 = {"subtask_id": "subtask_3", "instruction": debate_instruction2, "context": ["user query", "thinking2", "answer2"], "agent_collaboration": "Debate"}
    for r in range(rounds):
        for agent in debate_agents2:
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction2, r, is_sub_task=True)
            else:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1], debate_instruction2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent2(
        [taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc2['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs