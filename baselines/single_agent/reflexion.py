import inspect

# %%%%%%%%%%%%%%%%%%%% relexion (generator-evaluator) %%%%%%%%%%%%%%%%%%%%
async def forward(self, taskInfo):
    # Instruction for initial reasoning
    cot_initial_instruction = self.cot_instruction

    # Instruction for reflecting on previous attempts and feedback to improve
    cot_reflect_instruction = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_agent = LLMAgentBase(['thinking', 'answer'], 'Chain-of-Thought Agent', model=self.node_model, temperature=0.0)

    # Instruction for providing feedback and correcting the answer
    critic_instruction = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
    critic_agent = LLMAgentBase(['feedback', 'correct'], 'Critic Agent', model=self.node_model, temperature=0.0)
    
    N_max = self.max_round # Maximum number of attempts
    
    # Initial attempt
    cot_inputs = [taskInfo]
    thinking, answer = await cot_agent(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        # Get feedback and correct status from the critic
        feedback, correct = await critic_agent([taskInfo, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break
            
        # Add feedback to the inputs for the next iteration
        cot_inputs.extend([thinking, answer, feedback])

        # Reflect on previous attempts and refine the answer
        thinking, answer = await cot_agent(cot_inputs, cot_reflect_instruction, i + 1)

    final_answer = await self.make_final_answer(thinking, answer)

    return final_answer


func_string = inspect.getsource(forward)


Reflexion = {
    "thought": "To enhance its performance, an LLM can iteratively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution.",
    "name": "Self-Refine (Reflexion)",
    "code": """{func_string}""".format(func_string=func_string)
}



