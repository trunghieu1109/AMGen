async def forward_15(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Stage 0 - Sub-task 1: Extract and define variables using SC_CoT
    cot_sc_instruction = (
        "Sub-task 1: Extract the given set sizes |D|, |G|, |S|, |C|, total population and "
        "introduce variables S2 = sum of pairwise intersections, S3 = sum of triple intersections, x4 = |D∩G∩S∩C|."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                model=self.node_model, temperature=0.5)
                  for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                           model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 1: Synthesize and choose the most consistent extraction and definitions.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 1 - Sub-task 2: Formulate equations using Debate
    debate_instr = (
        "Sub-task 2: Formulate the equations for exactly-two and exactly-three overlaps: "
        "exactly_two = S2 - 3·S3 - 6·x4 = 437, exactly_three = S3 - 4·x4 = 234."
        " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent",
                                  model=self.node_model, role=role, temperature=0.5)
                     for role in self.debate_role]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instr,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking2, answer2 = await agent(
                    [taskInfo, thinking1, answer1], debate_instr, r, is_sub_task=True
                )
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(inputs, debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                           model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide the two equations.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 2 - Sub-task 3: Solve for x4 using SC_CoT
    cot_sc_instruction3 = (
        "Sub-task 3: Apply inclusion–exclusion: 900 = (|D|+|G|+|S|+|C|) – S2 + S3 – x4, "
        "substitute S2 and S3 from previous equations and solve for x4."
    )
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent",
                                 model=self.node_model, temperature=0.5)
                   for _ in range(self.max_sc)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents3:
        thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent",
                                           model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3,
        "Sub-task 3: Synthesize and choose the correct value of x4.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs