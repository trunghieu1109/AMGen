async def forward_7(self, taskInfo):
    from collections import Counter

    sub_tasks = []
    agents = []

    # Sub-task 1: Transform log_x(y^x) = 10
    cot_instruction_1 = "Transform log_x(y^x) = 10 into an equation in terms of x and y."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, transforming equation, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Transform log_y(x^4y) = 10
    cot_instruction_2 = "Transform log_y(x^4y) = 10 into an equation in terms of x and y."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, transforming equation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 3: Solve equation from Subtask 1 for one variable
    cot_instruction_3 = "Solve the equation from Subtask 1 for one variable in terms of the other."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, solving for one variable, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Solve equation from Subtask 2 for one variable
    cot_instruction_4 = "Solve the equation from Subtask 2 for one variable in terms of the other."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, solving for one variable, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Substitute expression from Subtask 3 into equation from Subtask 4
    reflexion_instruction_5 = "Substitute the expression from Subtask 3 into the equation from Subtask 4."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3, thinking4, answer4], reflexion_instruction_5, is_sub_task=True)
    agents.append(f"Reflexion agent {cot_agent_5.id}, substituting and finding relations, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Refine and validate relations from Subtask 5
    critic_agent_6 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    for _ in range(self.max_round):
        feedback6, correct6 = await critic_agent_6([taskInfo, thinking5, answer5], "please validate the relations and solutions.", is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_6.id}, validating relations, feedback: {feedback6.content}; correct: {correct6.content}")
        if correct6.content == "True":
            break

    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    # Sub-task 7: Calculate the value of xy
    debate_instruction_7 = "Use validated relations to compute the value of xy."
    debate_agents_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking7 = []
    all_answer7 = []
    for agent in debate_agents_7:
        thinking7, answer7 = await agent([taskInfo, thinking5, answer5], debate_instruction_7, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, computing xy, thinking: {thinking7.content}; answer: {answer7.content}")
        all_thinking7.append(thinking7)
        all_answer7.append(answer7)

    final_decision_agent_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await final_decision_agent_7([taskInfo] + all_thinking7 + all_answer7, "Make final decision for xy.", is_sub_task=True)
    agents.append(f"Final Decision agent, computing xy, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Subtask 7 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking7, answer7, sub_tasks, agents)
    return final_answer