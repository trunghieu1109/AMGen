async def forward_188(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: SC_CoT for collecting and summarizing physical origins
    cot_sc_instruction = (
        "Sub-task 1: Collect and summarize the physical origin and context of each effective particle—magnon, skyrmion, pion, and phonon.")
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                  for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, collecting origins, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings.append(thinking1)
        possible_answers.append(answer1)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1(
        [taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 1: Synthesize the most consistent and accurate summary of origins.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Debate to identify which are Goldstone or topological modes
    debate_instruction_3 = (
        "Sub-task 3: Evaluate which of these particles arise as Nambu–Goldstone modes or topological excitations tied directly to a spontaneously broken symmetry."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                       for role in self.debate_role]
    rounds = self.max_round
    all_thinking3 = [[] for _ in range(rounds)]
    all_answer3 = [[] for _ in range(rounds)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "Debate"
    }
    for r in range(rounds):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking1, answer1], debate_instruction_3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(inputs, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = (
        "Sub-task 3: Synthesize which are NG modes or topological excitations."
        "Given all the above thinking and answers, reason over them carefully and provide a final answer.")
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, thinking1, answer1] + all_thinking3[-1] + all_answer3[-1],
        final_instr3,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: Debate to choose the particle not tied to symmetry breaking
    debate_instruction_4 = (
        "Sub-task 4: Determine and select the single effective particle that is not associated with spontaneous symmetry breaking."
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                       for role in self.debate_role]
    all_thinking4 = [[] for _ in range(rounds)]
    all_answer4 = [[] for _ in range(rounds)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(rounds):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_4, r, is_sub_task=True)
            else:
                inputs4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(inputs4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = (
        "Sub-task 4: Synthesize and choose the single particle not tied to symmetry breaking."
        "Given all the above thinking and answers, reason over them carefully and provide a final answer.")
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1],
        final_instr4,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs