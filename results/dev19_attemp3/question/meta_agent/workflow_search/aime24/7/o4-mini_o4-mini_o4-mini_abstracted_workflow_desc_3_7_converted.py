async def forward_7(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Sub-task 1: SC_CoT
    cot_sc_instruction = (
        "Sub-task 1: Derive y^x = x^10 from log_x(y^x)=10 and x^{4y}=y^{10} from log_y(x^{4y})=10, showing each step.")
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings + possible_answers,
        "Sub-task 1: Synthesize and choose the most consistent answers for the exponential relations.", is_sub_task=True)
    sub_tasks.append(f"thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1, "answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Sub-task 2: Debate
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = "Sub-task 2: Solve y = x^{10/x}, substitute into x^{4y}=y^{10}, and derive the single-variable equation x^{(10/x)+1}=25." + debate_instr
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking = [[] for _ in range(self.max_round)]
    all_answer = [[] for _ in range(self.max_round)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":debate_instruction,"context":["user query",thinking1.content,answer1.content],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents:
            if r==0:
                inp = [taskInfo, thinking1, answer1]
            else:
                inp = [taskInfo, thinking1, answer1] + all_thinking[r-1] + all_answer[r-1]
            thinking, answer = await agent(inp, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking[-1] + all_answer[-1],
        "Sub-task 2: Given all above thinking and answers, reason over them carefully and provide the final derived equation.", is_sub_task=True)
    sub_tasks.append(f"thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2, "answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Sub-task 3: Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3: From x^{(10/x)+1}=25, extract x^{10/x}=25/x and hence y=25/x, then compute xy. " + reflect_inst
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_reflect_instruction,"context":["user query",thinking1.content,answer1.content,thinking2.content,answer2.content],"agent_collaboration":"Reflexion"}
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3],
            "Please review the answer above and criticize where it might be wrong. If correct output exactly 'True'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refinement thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3, "answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Sub-task 4: CoT verification
    cot4_instruction = "Sub-task 4: Verify that x>1, y>1 with y=25/x satisfies both original logarithmic equations and confirm uniqueness."  
    cot4_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot4_instruction,"context":["user query",thinking3.content,answer3.content],"agent_collaboration":"CoT"}
    thinking4, answer4 = await cot4_agent([taskInfo, thinking3, answer3], cot4_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot4_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4, "answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs