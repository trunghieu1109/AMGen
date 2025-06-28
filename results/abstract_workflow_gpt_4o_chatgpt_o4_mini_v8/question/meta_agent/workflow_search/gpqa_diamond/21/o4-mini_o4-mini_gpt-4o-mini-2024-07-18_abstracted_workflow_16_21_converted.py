async def forward_21(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Sub-task 1: parse student's query for key elements
    cot1_instruction = "Sub-task 1: Parse the student’s query to extract key elements: thermodynamic assessment in basic solution and kinetic assessment in acidic solution, plus the provided choices."
    cot1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot1([taskInfo], cot1_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot1.id}, parsing query, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: transform into specific evaluation tasks with self-consistency
    sc_instruction = "Sub-task 2: Transform the parsed query into two evaluation tasks: (1) thermodynamic oxidant strength of O₂ in basic solution, (2) kinetic reaction rate of O₂ in acidic solution; retain the four multiple-choice options."
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    answers = []
    thinking_map = {}
    answer_map = {}
    for agent in sc_agents:
        t2, a2 = await agent([taskInfo, thinking1, answer1], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, transforming tasks, thinking: {t2.content}; answer: {a2.content}")
        answers.append(a2.content)
        thinking_map[a2.content] = t2
        answer_map[a2.content] = a2
    majority = Counter(answers).most_common(1)[0][0]
    thinking2 = thinking_map[majority]
    answer2 = answer_map[majority]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: determine thermodynamic oxidizing strength
    cot3_instr = "Sub-task 3: Determine the standard reduction potential E° for O₂/H₂O ↔ OH⁻ in basic solution and compare to typical oxidants to classify O₂ as stronger or weaker oxidant."
    cot3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot3([taskInfo, thinking2, answer2], cot3_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot3.id}, thermodynamic analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: assess kinetic behavior in acidic solution
    cot4_instr = "Sub-task 4: Assess kinetic behavior of O₂ reduction in acidic solution, considering overpotential and electron-transfer barriers to classify reaction rate as faster or slower."
    cot4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot4([taskInfo, thinking2, answer2], cot4_instr, is_sub_task=True)
    agents.append(f"CoT agent {cot4.id}, kinetic analysis, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: integrate and select correct choice via debate
    debate_instr = "Sub-task 5: Integrate thermodynamic and kinetic classifications and choose the correct combination from the four options: weaker–faster, stronger–slower, weaker–slower, stronger–faster."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    rounds = self.max_round
    all_t = [[] for _ in range(rounds)]
    all_a = [[] for _ in range(rounds)]
    for r in range(rounds):
        for agent in debate_agents:
            inputs = [taskInfo, thinking3, answer3, thinking4, answer4]
            if r > 0:
                inputs += all_t[r-1] + all_a[r-1]
            t5, a5 = await agent(inputs, debate_instr, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integration and choice, thinking: {t5.content}; answer: {a5.content}")
            all_t[r].append(t5)
            all_a[r].append(a5)
    final_decision = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision([taskInfo] + all_t[-1] + all_a[-1], "Sub-task 5: Make final choice among the provided options.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision.id}, synthesizing choice, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer